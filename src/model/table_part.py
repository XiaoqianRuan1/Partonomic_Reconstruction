from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from toolz import valfilter

import pytorch3d
import numpy as np
from pytorch3d.loss import mesh_laplacian_smoothing as laplacian_smoothing
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import list_to_packed, list_to_padded, padded_to_list
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import os
from PIL import Image
#import trimesh

from .attention import DETR,Transformer,TextureTransformer
from .weights import ProgressiveWeightsField
from .encoder import Encoder,PartEncoder,AttEncoder
from .field import ProgressiveField,TextureField
from .generator import ProgressiveGenerator
from .loss import get_loss
from .renderer import Renderer, save_mesh_as_gif
from .tools import create_mlp, init_rotations, convert_3d_to_uv_coordinates, safe_model_state_dict, N_UNITS, N_LAYERS
from .tools import azim_to_rotation_matrix, elev_to_rotation_matrix, roll_to_rotation_matrix, cpu_angle_between
from utils import path_mkdir, use_seed
from utils.image import convert_to_img
from utils.logger import print_warning
from utils.mesh import save_mesh_as_obj, repeat, get_icosphere, normal_consistency, normalize
from utils.metrics import MeshEvaluator, ProxyEvaluator, PartEvaluator
from utils.pytorch import torch_to
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

# POSE & SCALE DEFAULT
N_POSES = 6
N_ELEV_AZIM = [1, 6]
SCALE_ELLIPSE = [1, 0.7, 0.7]
PRIOR_TRANSLATION = [0., 0., 2.732]

# NEIGHBOR REC LOSS DEFAULT (previously called swap loss)
MIN_ANGLE = 10
N_VPBINS = 5
MEMSIZE = 1024

N_K = 3
torch.autograd.set_detect_anomaly(True)

color_map = [
    [255,0,0],
    [0,0,255],
    [0,255,0],
    #[200,0,255],
    [255,255,255],
]
HIDDEN_DIM = 128
DROPOUT = 0.1
NHEADS =4 #8
DIM_FEEDFORWARD = 128 #128 #2048
ENC_LAYERS = 1
DEC_LAYERS = 1
PRE_NORM = False
DEEP_SUPERVISION = False
NUMBER_OF_QUERIES = 642
NEED_ENCODER=False


class PartMesh(Meshes):
    def __init__(self,verts,faces,labels,textures=None):
        """
        add a new attribute, labels
        """
        super().__init__(verts,faces)
        self.vert_labels = labels # similar with verts_features_padded
        self.verts = verts
        self.faces = faces
        self._N = len(labels)
        self._vert_labels_list = None
        self._vert_labels_packed = None
        self._vert_labels_padded = None
        max_F = labels[0].shape[0]
        self.num_labels_per_mesh = [max_F] * self._N
        if textures != None:
            self.textures = textures
        if isinstance(labels, list):
            self._vert_labels_list = labels
        elif torch.is_tensor(labels):
            self._vert_labels_padded = labels
        else:
            raise ValueError("Vert labels are incorrect.")
    
    def vert_labels_list(self):
        if self._vert_labels_list is None:
            assert (
                self._vert_labels_padded is not None
            ), "vert_labels_padded is required to compute verts_list."
            self._vert_labels_list = padded_to_list(self._vert_labels_padded, split_size=self.num_labels_per_mesh)
        return self._vert_labels_list
    
    def vert_labels_padded(self, refresh=False):
        if not (
            refresh or any(v is None for v in [self._vert_labels_padded])
        ):
            return
        vert_labels_list = self.vert_labels_list()
        if self.isempty():
            self._vert_labels_padded = torch.zeros(
                (self._N, 0, 3), dtype=torch.float32, device=self.device
            )
        else:
            self._vert_labels_padded = list_to_padded(
                vert_labels_list, (self._V, 3), pad_value=0.0, equisized=self.equisized
            )
    
    def verts_labels_packed(self, refresh=False):
        if not (
            refresh
            or any(
                v is None
                for v in [
                    self._vert_labels_packed,
                ]
            )
        ):
            return
        verts_labels_list = self.vert_labels_list()
        verts_labels_list_to_packed = list_to_packed(verts_labels_list)
        self._verts_labels_packed = verts_labels_list_to_packed[0]
        return self._verts_labels_packed
        
    def update_padded(self, new_verts_padded):
        """
        This function is the extension of original update_padded function with new attribute, vert_labels.
        Args: new_points_padded: FloatTensor of shape (N,V,3)
        Returns: Meshes with updated padded representations
        """
        def check_shapes(x,size):
            if x.shape[0]!=size[0]:
                raise ValueError("new values must have the same batch dimension.")
            if x.shape[1]!=size[1]:
                raise ValueError("new values must have the same number of points.")
            if x.shape[2]!=size[2]:
                raise ValueError("new values must have the same dimension.")
        check_shapes(new_verts_padded, [self._N,self._V,3])
        new = self.__class__(verts=new_verts_padded,faces=self.faces_padded(),labels=self.vert_labels)
        if new._N != self._N or new._V != self._V or new._F != self._F:
            raise ValueError("Inconsistent sizes after construction.")
        new.equisized = self.equisized
        new.textures = self.textures
        copy_tensors = ['_num_verts_per_mesh','_num_faces_per_mesh','valid']
        for k in copy_tensors:
            v = getattr(self,k)
            if torch.is_tensor(v):
                setattr(new,k,v)
        new._faces_list = self._faces_list
        if self._verts_packed is not None:
            copy_tensors = [
                "_faces_packed",
                "_verts_packed_to_mesh_idx",
                "_faces_packed_to_mesh_idx",
                "_mesh_to_verts_packed_first_idx",
                "_mesh_to_faces_packed_first_idx",
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy
            # update verts_packed
            pad_to_packed = self.verts_padded_to_packed_idx()
            new_verts_packed = new_verts_padded.reshape(-1, 3)[pad_to_packed, :]
            new._verts_packed = new_verts_packed
            new._verts_padded_to_packed_idx = pad_to_packed

        # update edges packed if they are computed in self
        if self._edges_packed is not None:
            copy_tensors = [
                "_edges_packed",
                "_edges_packed_to_mesh_idx",
                "_mesh_to_edges_packed_first_idx",
                "_faces_packed_to_edges_packed",
                "_num_edges_per_mesh",
            ]
            for k in copy_tensors:
                v = getattr(self, k)
                assert torch.is_tensor(v)
                setattr(new, k, v)  # shallow copy

        # update laplacian if it is compute in self
        if self._laplacian_packed is not None:
            new._laplacian_packed = self._laplacian_packed

        assert new._verts_list is None
        assert new._verts_normals_packed is None
        assert new._faces_normals_packed is None
        assert new._faces_areas_packed is None

        return new

    def detach(self):
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        labels = self.vert_labels
        new_verts_list = [v.detach() for v in verts_list]
        new_faces_list = [f.detach() for f in faces_list]
        new_labels = labels.detach()
        other = self.__class__(verts=new_verts_list,faces=new_faces_list,labels=new_labels)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self,k)
            if torch.is_tensor(v):
                setattr(other,k,v.detach())
        if self.textures is not None:
            other.textures = self.textures.detach()
        return other

    def __getitem__(self, index):
        """
        an extension of Meshes for PartMesh
        """
        if isinstance(index, (int, slice)):
            verts = self.verts_list()[index]
            faces = self.faces_list()[index]
        elif isinstance(index, list):
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            # NOTE consider converting index to cpu for efficiency
            if index.dtype == torch.bool:
                # advanced indexing on a single dimension
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            verts = [self.verts_list()[i] for i in index]
            faces = [self.faces_list()[i] for i in index]
        else:
            raise IndexError(index)

        labels = self.vert_labels
        textures = None if self.textures is None else self.textures[index]

        if torch.is_tensor(verts) and torch.is_tensor(faces):
            return self.__class__(verts=[verts], faces=[faces], labels=labels,textures=textures)
        elif isinstance(verts, list) and isinstance(faces, list):
            return self.__class__(verts=verts, faces=faces, labels=labels, textures=textures)
        else:
            raise ValueError("(verts, faces) not defined correctly")

    def clone(self):
        verts_list = self.verts_list()
        faces_list = self.faces_list()
        labels_list = self.vert_labels
        new_verts_list = [v.clone() for v in verts_list]
        new_faces_list = [f.clone() for f in faces_list]
        new_labels = [l.clone() for l in labels_list]
        other = self.__class__(verts=new_verts_list,faces=new_faces_list,labels=new_labels)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())

        # Textures is not a tensor but has a clone method
        if self.textures is not None:
            other.textures = self.textures.clone()
        return other

    def extend(self, N: int):
        """
        Create new Meshes class which contains each input mesh N times

        Args:
            N: number of new copies of each mesh.

        Returns:
            new Meshes object.
        """
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
        if N <= 0:
            raise ValueError("N must be > 0.")
        new_verts_list, new_faces_list = [], []
        new_labels = []
        for verts, faces, labels in zip(self.verts_list(), self.faces_list(),self.vert_labels):
            new_verts_list.extend(verts.clone() for _ in range(N))
            new_faces_list.extend(faces.clone() for _ in range(N))
            new_labels.extend(labels.clone() for _ in range(N))

        tex = None
        if self.textures is not None:
            tex = self.textures.extend(N)

        return self.__class__(verts=new_verts_list, faces=new_faces_list, labels=new_labels, textures=tex)

def repeat_part(mesh,N):
    """
    Return N copies, an extension of Meshes, applied for PartMeshes
    """
    assert N>=1
    if N==1:
        return mesh
    new_verts_list, new_faces_list = [], []
    for _ in range(N):
        new_verts_list.extend(verts.clone() for verts in mesh.verts_list())
        new_faces_list.extend(faces.clone() for faces in mesh.faces_list())
    labels = mesh.vert_labels
    new_labels = labels.repeat(N,1,1)
    return PartMesh(verts=new_verts_list,faces=new_faces_list,labels=new_labels)

class Unicorn(nn.Module):
    name = 'unicorn'
    def __init__(self, img_size, **kwargs):
        super().__init__()
        self.init_kwargs = deepcopy(kwargs)
        self.init_kwargs['img_size'] = img_size
        self._init_encoder(img_size, **kwargs.get('encoder', {}))
        self._init_transformer(N_K)
        self._init_shapes(**kwargs.get('shape',{}))
        self._init_meshes(**kwargs.get('mesh', {}))
        self.layer_renderer = Renderer(img_size, True, False, **kwargs.get('renderer', {}))
        self.part_renderer = Renderer(img_size, False, True, **kwargs.get('renderer', {}))
        self._init_rend_predictors(**kwargs.get('rend_predictor', {}))
        self._init_background_model(img_size, **kwargs.get('background', {}))
        self._init_milestones(**kwargs.get('milestones', {}))
        self._init_loss(**kwargs.get('loss', {}))
        self.prop_heads = torch.zeros(self.n_poses)
        self.cur_epoch, self.cur_iter = 0, 0
        self._debug = False

    @property
    def n_features(self):
        return self.encoder.out_ch if self.shared_encoder else self.encoder_sh.out_ch
       
    @property
    def tx_features(self):
        return self.local_encoder.out_ch

    @property
    def tx_code_size(self):
        return self.txt_field.current_code_size

    @property
    def sh_code_size(self):
        return self.direct_field.current_code_size

    def _init_encoder(self, img_size, **kwargs):
        """
        different encoders for different parts;
        shape encoder is changed, with the original n_feature to the k*n_feature;
        """
        self.shared_encoder = kwargs.pop('shared', False)
        if self.shared_encoder:
            self.encoder = Encoder(img_size, **kwargs)
            self.local_encoder = AttEncoder(img_size, **kwargs)
        else:
            self.encoder_sh = Encoder(img_size,**kwargs)
            self.local_encoder = AttEncoder(img_size, **kwargs)
            self.encoder_sc = Encoder(img_size,**kwargs)
            #self.encoder_tx = Encoder(img_size,**kwargs)
            self.encoder_pose = Encoder(img_size,**kwargs)
            if len(self.init_kwargs.get('background', {})) > 0:
                self.encoder_bg = Encoder(img_size, **kwargs)
            
    def _init_transformer(self,N_K):
        self.transformer_part = DETR(N_K)
        self.transformer_txt = TextureTransformer(1024) 
    
    def _init_shapes(self,**kwargs):
        kwargs = deepcopy(kwargs)
        mesh_init = kwargs.pop('init','sphere')
        scale = kwargs.pop('scale',1)
        if 'sphere' in mesh_init or 'ellipse' in mesh_init:
            mesh = get_icosphere(3 if 'hr' in mesh_init else 2)
            if 'ellipse' in mesh_init:
                scale = scale * torch.Tensor([SCALE_ELLIPSE])
        else:
            raise NotImplementedError
        self.mesh_src = mesh.scale_verts(scale)
        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))  # retrocompatibility
        subdivide = SubdivideMeshes()
        self.mesh_up = subdivide(self.mesh_src)
        
        # two symmetric assumptions
        coarse_vertices = self.mesh_src.get_mesh_verts_faces(0)[0]
        coarse_center_flip = coarse_vertices.clone()
        coarse_center_flip[:,2] *= -1
        self.coarse_index = torch.cdist(coarse_vertices, coarse_center_flip).min(1)[1]
        
        vertices = self.mesh_up.get_mesh_verts_faces(0)[0]
        vertex_center_flip = vertices.clone()
        vertex_center_flip[:,2] *= -1
        self.flip_index = torch.cdist(vertices, vertex_center_flip).min(1)[1]
        
        direct_kwargs = kwargs.pop('direct_field',{})
        weights_kwargs = kwargs.pop('weights_field',{})

        self.direct_field = ProgressiveField(inp_dim=self.n_features, name='direction', **direct_kwargs)
        self.weights_field = ProgressiveWeightsField(inp_dim=self.n_features, name='weights',**weights_kwargs)

    def _init_meshes(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.register_buffer('uvs', convert_3d_to_uv_coordinates(self.mesh_up.get_mesh_verts_faces(0)[0])[None]) # only set the size;

        self.use_mean_txt = kwargs.pop('use_mean_txt', kwargs.pop('use_mean_text', False))  # retrocompatibility 
        tgen_kwargs = kwargs.pop('texture_uv', {})
        deform_kwargs = kwargs.pop('deform_field',{})
        assert len(kwargs) == 0
        self.txt_field = TextureField(inp_dim=self.tx_features, name='texture', **tgen_kwargs)
        self.deform_field1 = ProgressiveField(inp_dim=self.tx_features, name='deform1', **deform_kwargs)
        self.deform_field2 = ProgressiveField(inp_dim=self.tx_features, name='deform2', **deform_kwargs)
        self.deform_field3 = ProgressiveField(inp_dim=self.tx_features, name='deform3', **deform_kwargs)
        #self.deform_field4 = ProgressiveField(inp_dim=self.tx_features, name='deform4', **deform_kwargs)

    def _init_rend_predictors(self, **kwargs): # pose
        kwargs = deepcopy(kwargs)
        self.n_poses = kwargs.pop('n_poses', N_POSES)
        n_elev, n_azim = kwargs.pop('n_elev_azim', N_ELEV_AZIM)
        assert self.n_poses == n_elev * n_azim
        self.alternate_optim = kwargs.pop('alternate_optim', True)
        self.pose_step = True

        NF, NP = self.n_features, self.n_poses
        NU, NL = kwargs.pop('n_reg_units', N_UNITS), kwargs.pop('n_reg_layers', N_LAYERS)

        # Translation
        self.T_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        T_range = kwargs.pop('T_range', 1)
        T_range = [T_range] * 3 if isinstance(T_range, (int, float)) else T_range
        self.register_buffer('T_range', torch.Tensor(T_range))
        self.register_buffer('T_init', torch.Tensor(kwargs.pop('prior_translation', PRIOR_TRANSLATION)))

        # Rotation
        self.rot_regressors = nn.ModuleList([create_mlp(NF, 3, NU, NL, zero_last_init=True) for _ in range(NP)])
        a_range, e_range, r_range = kwargs.pop('azim_range'), kwargs.pop('elev_range'), kwargs.pop('roll_range')
        azim, elev, roll = [(e[1] - e[0]) / n for e, n in zip([a_range, e_range, r_range], [n_azim, n_elev, 1])]
        R_init = init_rotations('uniform', n_elev=n_elev, n_azim=n_azim, elev_range=e_range, azim_range=a_range)
        # In practice we extend the range a bit to allow overlap in case of multiple candidates
        if self.n_poses == 1:
            self.register_buffer('R_range', torch.Tensor([azim * 0.5, elev * 0.5, roll * 0.5]))
        else:
            self.register_buffer('R_range', torch.Tensor([azim * 0.52, elev * 0.52, roll * 0.52]))
        self.register_buffer('R_init', R_init)
        self.azim_range, self.elev_range, self.roll_range = a_range, e_range, r_range

        # Scale
        self.scale_regressor = create_mlp(NF, 3, NU, NL, zero_last_init=True)
        scale_range = kwargs.pop('scale_range', 0.5)
        scale_range = [scale_range] * 3 if isinstance(scale_range, (int, float)) else scale_range
        self.register_buffer('scale_range', torch.Tensor(scale_range))
        self.register_buffer('scale_init', torch.ones(3))
        
        """
        self.scale_coarse = create_mlp(NF, 3, NU, NL, zero_last_init=True)
        coarse_range = kwargs.pop('coarse_range', 0.5)
        coarse_range = [coarse_range] * 3 if isinstance(coarse_range, (int, float)) else coarse_range
        self.register_buffer('coarse_range', torch.Tensor(coarse_range))
        self.register_buffer('coarse_init', torch.ones(3))
        """

        # Pose probabilities
        if NP > 1:
            self.proba_regressor = create_mlp(NF, NP, NU, NL)

        assert len(kwargs) == 0, kwargs

    @property
    def n_candidates(self):
        return 1 if self.hard_select else self.n_poses

    @property
    def hard_select(self):
        if self.alternate_optim and not self._debug:
            return False if (self.training and self.pose_step) else True
        else:
            return False

    def _init_background_model(self, img_size, **kwargs):
        if len(kwargs) > 0:
            bkg_kwargs = deepcopy(kwargs)
            self.bkg_generator = ProgressiveGenerator(inp_dim=self.n_features, img_size=img_size, **bkg_kwargs)

    def _init_milestones(self, **kwargs):
        kwargs = deepcopy(kwargs)
        self.milestones = {
            'constant_txt': kwargs.pop('constant_txt', kwargs.pop('contant_text', 0)),  # retrocompatibility
            'freeze_T_pred': kwargs.pop('freeze_T_predictor', 0),
            'freeze_R_pred': kwargs.pop('freeze_R_predictor', 0),
            'freeze_s_pred': kwargs.pop('freeze_scale_predictor', 0),
            'freeze_shape': kwargs.pop('freeze_shape', 0),
            'mean_txt': kwargs.pop('mean_txt', kwargs.pop('mean_text', self.use_mean_txt)),  # retrocompatibility
        }
        assert len(kwargs) == 0

    def _init_loss(self, **kwargs):
        kwargs = deepcopy(kwargs)
        loss_weights = {
            #'entropy': kwargs.pop('entropy_weight',0.1),
            #'union': kwargs.pop('union_weight',0.01),
            'rgb': kwargs.pop('rgb_weight', 1.0),
            'mask': kwargs.pop('mask_weight', 0),
            'parts': kwargs.pop('part_weight', 0),
            'flip': kwargs.pop('flip_weight',0),
            'normal': kwargs.pop('normal_weight', 0),
            'laplacian': kwargs.pop('laplacian_weight', 0),
            'perceptual': kwargs.pop('perceptual_weight', 0),
            'uniform': kwargs.pop('uniform_weight', 0),
            'neighbor': kwargs.pop('neighbor_weight', kwargs.pop('swap_weight', 0)),  # retrocompatibility
        }
        name = kwargs.pop('name', 'mse')
        mask = kwargs.pop('mask','mask')
        perceptual_kwargs = kwargs.pop('perceptual', {})
        self.nbr_memsize = kwargs.pop('nbr_memsize', kwargs.pop('swap_memsize', MEMSIZE))  # retro
        self.nbr_n_vpbins = kwargs.pop('nbr_n_vpbins', kwargs.pop('swap_n_vpbins', N_VPBINS))  # retro
        self.nbr_min_angle = kwargs.pop('nbr_min_angle', kwargs.pop('swap_min_angle', MIN_ANGLE))  # retro
        self.nbr_memory = {k: torch.empty(0) for k in ['sh', 'tx', 'S', 'R', 'T', 'bg', 'img', 'mask']}
        kwargs.pop('swap_equal_bins', False)  # retro
        assert len(kwargs) == 0, kwargs

        self.loss_weights = valfilter(lambda v: v > 0, loss_weights)
        self.loss_names = [f'loss_{n}' for n in list(self.loss_weights.keys()) + ['total']]
        self.criterion = get_loss(name)(reduction='none')
        self.mask_loss = get_loss(mask)(eps=1e-6, reduction='none')
        if 'perceptual' in self.loss_weights:
            self.perceptual_loss = get_loss('perceptual')(**perceptual_kwargs)

    @property
    def pred_background(self):
        return hasattr(self, 'bkg_generator')

    def is_live(self, name):
        milestone = self.milestones[name]
        if isinstance(milestone, bool):
            return milestone
        else:
            return True if self.cur_epoch < milestone else False

    def to(self, device):
        super().to(device)
        self.mesh_src = self.mesh_src.to(device)
        self.layer_renderer = self.layer_renderer.to(device)
        self.part_renderer = self.part_renderer.to(device)
        self.mesh_up = self.mesh_up.to(device)
        return self

    def forward(self, inp, debug=False):
        # XXX pytorch3d objects are not well handled by DDP so we need to manually move them to GPU
        # self.mesh_src, self.renderer = [t.to(inp['imgs'].device) for t in [self.mesh_src, self.renderer]]
        self._debug = debug
        imgs, K, B = inp['imgs'], self.n_candidates, len(inp['imgs'])
        masks = inp['masks']
        parts = inp['parts']
        perturbed = self.training and np.random.binomial(1, p=0.2)
        average_txt = self.is_live('constant_txt') or (perturbed and self.use_mean_txt and self.is_live('mean_txt'))
        # average_txt = 0, parameters of milestones in config file
        meshes, meshes_part, (R, T), bkgs = self.predict_mesh_pose_bkg(imgs, average_txt)
        if self.alternate_optim:
            if self.pose_step:
                meshes, bkgs = meshes.detach(), bkgs.detach() if self.pred_background else None
                meshes_part = meshes_part.detach()
            else:
                R, T = R.detach(), T.detach()
        
        meshes_to_render = repeat(meshes, len(T) // len(meshes))
        fgs, alpha = self.layer_renderer(meshes_to_render, R, T).split([3, 1], dim=1)  # (K*B)CHW
        rec = fgs * alpha + (1 - alpha) * bkgs if self.pred_background else fgs
        parts_to_render = repeat_part(meshes_part, len(T) // len(meshes_part))
        rec_part = self.part_renderer(parts_to_render, R, T)
        
        #losses, select_idx = self.compute_losses(meshes, alpha, masks, rec, imgs, rec_part, R, T, average_txt=average_txt)
        losses, select_idx = self.compute_losses(meshes, alpha, masks, rec, imgs, rec_part, parts, R, T, average_txt=average_txt)
        
        if debug:
            out = rec.view(K, B, *rec.shape[1:]) if K > 1 else rec[None]
            self._debug = False
        else:
            rec = rec.view(K, B, *rec.shape[1:])[select_idx, torch.arange(B)] if K > 1 else rec
            out = losses, rec

        return out

    def predict_mesh_pose_bkg(self, imgs, average_txt=False):
        if self.shared_encoder:
            features = self.encoder(imgs) # resnet; 
            local_features = self.local_encoder(imgs)
            #features_part = self.transformer_part(local_features)
            #features_tx = self.transformer_txt(local_features)
            meshes = self.predict_meshes(features, local_features, average_txt=average_txt)
            part_meshes = self.predict_part(meshes, features)
            R, T = self.predict_poses(features)
            bkgs = self.predict_background(features) if self.pred_background else None
        else:
            features_sh = self.encoder_sh(imgs)
            local_features = self.local_encoder(imgs)
            #local_features, features_part = self.encoder_part(imgs)
            #features_part = self.transformer_part(local_features)
            #features_tx = self.transformer_txt(local_features)
            #features_tx = self.encoder_tx(imgs)
            features_sc = self.encoder_sc(imgs)
            meshes = self.predict_meshes(features_sh, local_features, features_sc,average_txt=average_txt)
            part_meshes = self.predict_part(meshes,features_sh)
            R, T = self.predict_poses(self.encoder_pose(imgs))
            bkgs = self.predict_background(self.encoder_bg(imgs)) if self.pred_background else None
        return meshes, part_meshes, (R, T), bkgs

    def predict_meshes(self,features_sh, local_features, features_sc=None, average_txt=False):
        if features_sc is None:
            features_sc = features_sh        
        meshes = self.predict_verts(features_sh)
        verts,faces = meshes.get_mesh_verts_faces(0)
        meshes.offset_verts_(self.predict_disp_verts(verts,local_features,features_sh))
        meshes.textures = self.predict_textures(faces,local_features,average_txt)
        meshes.scale_verts_(self.predict_scales(features_sc))
        return meshes
        
    def predict_part(self,meshes,features):
        weights = self.weights = self.subdivide_weights(features)
        verts, faces = meshes.verts_padded(), meshes.faces_padded()
        part_meshes = PartMesh(verts=verts,faces=faces,labels=weights)
        return part_meshes

    def predict_verts(self,features):
        """
        directly predict the shape deformation.
        The direct_prediction field and the scale field share the same features.
        Return:
          the upsampled meshes
        """
        faces, textures = self.mesh_src.faces_padded(), self.mesh_src.textures
        verts = self.mesh_src.get_mesh_verts_faces(0)[0]
        new_verts = self.direct_predict_verts(verts,features)
        vert_scales = self.predict_scales(features)[:, None]
        B = new_verts.shape[0]
        faces = faces.repeat(B,1,1)
        self.direct_verts = new_verts * vert_scales
        meshes = Meshes(self.direct_verts,faces)
        subdivide = SubdivideMeshes()
        meshes = subdivide(meshes)
        return meshes

    def direct_predict_verts(self,verts,features):
        verts = self.direct_field(verts,features) # BK3
        if self.is_live('freeze_shape'):
            verts = verts*0
        return verts
        
    def direct_predict_weights(self,verts,features):
        weights = self.weights_field(verts,features)
        weights = nn.functional.softmax(weights,dim=2)
        return weights

    def predict_disp_verts(self,verts,local_features,features_seg):
        features = self.transformer_part(local_features)
        disp_verts = []
        disp_verts.append(self.deform_field1(verts,features[:,0,:]))
        disp_verts.append(self.deform_field2(verts,features[:,1,:]))
        disp_verts.append(self.deform_field3(verts,features[:,2,:]))
        #disp_verts.append(self.deform_field4(verts,features[:,3,:]))
        disp_verts = torch.stack(disp_verts,1) # BKV3
        self.weights = self.subdivide_weights(features_seg)
        weights = self.weights.permute(0,2,1)
        weights = weights.unsqueeze(3).repeat(1,1,1,3)
        disp_verts = (disp_verts*weights).sum(1)
        if self.is_live('freeze_shape'):
            disp_verts = disp_verts * 0
        return disp_verts.view(-1, 3)

    def subdivide_weights(self, features):
        verts, edges = self.mesh_src.verts_padded(), self.mesh_src.edges_packed()
        vert = self.mesh_src.get_mesh_verts_faces(0)[0]
        weights = self.direct_predict_weights(vert,features)
        new_weights = weights[:, edges].mean(dim=2)
        new_weights = torch.cat([weights,new_weights],dim=1)
        return new_weights
    
    def entropy_weights_loss(self):
        loss = torch.mean(self.weights*torch.log2(self.weights+0.001))
        return loss

    def part_number_regularization(self):
        N,K = self.weights.shape[1], self.weights.shape[2]
        loss = torch.sum(self.weights,dim=1)-N/K
        loss = torch.mean(torch.abs(loss))
        return loss
    
    def segment_loss(self,part_pred,part_ground):
        """
        split the prediction and ground truth as the foreground and background
        """
        #loss = torch.mean(self.part_loss(part_pred,part_ground).sum(3))
        part_fgs, part_bgs = part_pred.split([N_K, 1],dim=1)
        ground_fgs, ground_bgs = torch.tensor(part_ground,dtype=torch.float32).split([N_K, 1],dim=-1)
        #loss = self.criterion(part_fgs, ground_fgs.permute(0,3,1,2).to(part_fgs.device))
        ground_fgs = ground_fgs.permute(0,3,1,2)
        #part_fgs = part_fgs
        loss = torch.mean(ground_fgs*torch.log2(part_fgs+0.001))
        return loss

    def vertex_flip(self, meshes):
        coarse_vertices = self.direct_verts.index_select(1, self.coarse_index.to(self.direct_verts.device))
        coarse_vertices[...,2] *= -1
        coarse_loss = (coarse_vertices - self.direct_verts).norm(dim=2).mean()
    
        disp_verts = meshes.verts_padded()
        vertices = disp_verts.index_select(1, self.flip_index.to(disp_verts.device))
        vertices[...,2] *= -1
        loss = (vertices - disp_verts).norm(dim=2).mean()
        return coarse_loss+loss

    def predict_textures(self, faces, features, average_txt=False):
        B = len(features)
        maps = self.transformer_txt(features)
        maps = self.txt_field(maps)
        if average_txt:
            H, W = maps.shape[-2:]
            nb = int(H * W * 0.2)
            idxh, idxw = torch.randperm(H)[:nb], torch.randperm(W)[:nb]
            maps = maps[:, :, idxh, idxw].mean(2)[..., None, None].expand(-1, -1, H, W)
        return TexturesUV(maps.permute(0, 2, 3, 1), faces[None].expand(B, -1, -1), self.uvs.expand(B, -1, -1))

    def predict_scales(self, features):
        s_pred = self.scale_regressor(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales = s_pred * self.scale_range + self.scale_init
        return self._scales

    """
    def predict_scales_coarse(self,features):
        s_pred = self.scale_coarse(features).tanh()
        if self.is_live('freeze_s_pred'):
            s_pred = s_pred * 0
        self._scales_coarse = s_pred * self.coarse_range + self.coarse_init
        return self._scales_coarse
    """

    def predict_poses(self, features):
        B = len(features)

        T_pred = torch.stack([p(features) for p in self.T_regressors], dim=0).tanh()
        if self.is_live('freeze_T_pred'):
            T_pred = T_pred * 0
        T = (T_pred * self.T_range + self.T_init).view(-1, 3)

        R_pred = torch.stack([p(features) for p in self.rot_regressors], dim=0)  # KBC
        R_pred = R_pred.tanh()[..., [1, 0, 2]]  # XXX for retro-compatibility
        if self.is_live('freeze_R_pred'):
            R_pred = R_pred * 0
        R_pred = (R_pred * self.R_range + self.R_init[:, None]).view(-1, 3)
        azim, elev, roll = map(lambda t: t.squeeze(1), R_pred.split([1, 1, 1], 1))
        R = azim_to_rotation_matrix(azim) @ elev_to_rotation_matrix(elev) @ roll_to_rotation_matrix(roll)

        if self.n_poses > 1:
            weights = self.proba_regressor(features.view(B, -1)).permute(1, 0)
            self._pose_proba = torch.softmax(weights, dim=0)  # KB
            if self.hard_select:
                indices = self._pose_proba.max(0)[1]
                select_fn = lambda t: t.view(self.n_poses, B, *t.shape[1:])[indices, torch.arange(B)]
                R, T = map(select_fn, [R, T])
        return R, T

    def predict_background(self, features):
        res = self.bkg_generator(features)
        return res.repeat(self.n_candidates, 1, 1, 1) if self.n_candidates > 1 else res

    def compute_losses(self, meshes, alpha, masks, rec, imgs, rec_part, parts, R, T, average_txt=False):
    #def compute_losses(self, meshes, alpha, masks, rec, imgs, rec_part, R, T, average_txt=False):
        K, B = self.n_candidates, len(imgs)
        if K > 1:
            imgs = imgs.repeat(K, 1, 1, 1)
            masks = masks.repeat(K, 1, 1, 1)
            parts = parts.repeat(K, 1, 1, 1)
        losses = {k: torch.tensor(0.0, device=imgs.device) for k in self.loss_weights}
        if self.training:
            update_3d, update_pose = (not self.pose_step, self.pose_step) if self.alternate_optim else (True, True)
        else:
            update_3d, update_pose = (False, False)

        #losses['entropy'] = self.loss_weights['entropy'] * self.entropy_weights_loss().abs().mean()
        # Standard reconstrution error on RGB values
        if 'rgb' in losses:
            #losses['rgb'] = self.loss_weights['rgb'] * self.criterion(rec*alpha, imgs*masks).flatten(1).mean(1) + self.loss_weights['rgb'] * self.criterion(refined_rec*refined_alpha, imgs*masks).flatten(1).mean(1)
            losses['rgb'] = self.loss_weights['rgb'] * self.criterion(rec*alpha, imgs*masks).flatten(1).mean(1)
        if 'mask' in losses:
            #losses['mask'] = self.loss_weights['mask'] * self.mask_loss(alpha, masks) + self.loss_weights['mask'] * self.mask_loss(refined_alpha, masks)
            losses['mask'] = self.loss_weights['mask'] * self.mask_loss(alpha, masks)
        
        if 'entropy' in losses:
            losses['entropy'] = self.loss_weights['entropy'] * self.entropy_weights_loss().abs().mean()
        if 'union' in losses:
            losses['union'] = self.loss_weights['union'] * self.part_number_regularization().abs().mean()
        
        if "parts" in losses:
            losses['parts'] = self.loss_weights['parts'] * self.segment_loss(rec_part,parts).abs().mean()
            #losses['parts'] = self.loss_weights['parts'] * self.segment_loss(rec_part,parts).abs().mean() + self.loss_weights['parts'] * self.segment_loss(refined_rec_part, parts).abs().mean()
        
        if 'flip' in losses:
            losses['flip'] = self.loss_weights['flip'] * self.vertex_flip(meshes)
        # Perceptual loss
        if 'perceptual' in losses:
            losses['perceptual'] = self.loss_weights['perceptual'] * self.perceptual_loss(rec,imgs) 
        # Mesh regularization
        if update_3d:
            if 'normal' in losses:
                losses['normal'] = self.loss_weights['normal'] * normal_consistency(meshes) 
            if 'laplacian' in losses:
                losses['laplacian'] = self.loss_weights['laplacian'] * laplacian_smoothing(meshes, method='uniform') 

        # Neighbor reconstruction loss
        # XXX when latent spaces are small, codes are similar so there is no need to compute the loss
        if update_3d and 'neighbor' in losses and (self.tx_code_size > 0 and self.sh_code_size > 0):
            B, dev = len(meshes), imgs.device
            verts, faces, textures = meshes.verts_padded(), meshes.faces_padded(), meshes.textures
            scales = self._scales[:, None]
            #z_dh,z_sh,z_tx = [m._latent for m in [self.direct_field, self.deform_field,self.txt_generator]]
            z_sh,z_tx = [m._latent for m in [self.direct_field, self.txt_field]]
            #z_sh1, z_sh2, z_sh3, z_sh4 = [m._latent for m in [self.deform_field1, self.deform_field2, self.deform_field3, self.deform_field4]]
            #z_sh1, z_sh2, z_sh3 = [m._latent for m in [self.deform_field1, self.deform_field2, self.deform_field3]]
            # features: feature_coarse, shape_deformation, texture; # z_sh: k*N; 
            z_bg = self.bkg_generator._latent if self.pred_background else torch.empty(B, 1, device=dev)
            for n, t in [('sh',z_sh), ('tx', z_tx), ('bg', z_bg), ('S', scales), ('R', R), ('T', T), ('img', imgs),('mask', masks)]:
                self.nbr_memory[n] = torch.cat([self.nbr_memory[n].to(dev), t.detach()])[-self.nbr_memsize:]

            # we compute the nearest neighbors in random bins
            min_angle, nb_vpbins = self.nbr_min_angle, self.nbr_n_vpbins
            with torch.no_grad():
                #sim_dh = (z_dh[None] - self.nbr_memory['dh'][:, None]).pow(2).sum(-1)
                sim_sh = (z_sh[None] - self.nbr_memory['sh'][:, None]).pow(2).sum(-1)
                sim_tx = (z_tx[None] - self.nbr_memory['tx'][:, None]).view(self.nbr_memory['tx'].shape[0],z_tx.shape[0], -1).pow(2).sum(-1)
                """
                sim_sh1 = (z_sh1[None] - self.nbr_memory['sh1'][:, None]).pow(2).sum(-1)
                sim_sh2 = (z_sh2[None] - self.nbr_memory['sh2'][:, None]).pow(2).sum(-1)
                sim_sh3 = (z_sh3[None] - self.nbr_memory['sh3'][:, None]).pow(2).sum(-1)
                sim_sh4 = (z_sh4[None] - self.nbr_memory['sh4'][:, None]).pow(2).sum(-1)
                """
                #sim_tx = (z_tx[None] - self.nbr_memory['tx'][:, None]).pow(2).sum(-1)
                angles = cpu_angle_between(self.nbr_memory['R'][:, None], R[None]).view(sim_sh.shape)
                angle_bins = torch.randint(0, nb_vpbins, (B,), device=dev).float()
                # we create bins with equal angle range and sample from them
                bin_size = (180. - min_angle) / nb_vpbins  # we compute the size for uniform bins
                # invalid items are items whose angles are outside [min_angle, max_angle[
                min_angles, max_angles = [(angle_bins + k) * bin_size + min_angle for k in range(2)]
                invalid_mask = (angles < min_angles).float() + (angles >= max_angles).float()
                idx_sh,idx_tx = map(lambda t: (t + t.max() * invalid_mask).argmin(0), [sim_sh,sim_tx])
                #idx_sh1, idx_sh2, idx_sh3, idx_sh4 = map(lambda t: (t + t.max() * invalid_mask).argmin(0), [sim_sh1,sim_sh2,sim_sh3,sim_sh4])

            v_src, f_src = self.mesh_src.get_mesh_verts_faces(0)
            #v_src, f_src = meshes.get_mesh_verts_faces(0)
            nbr_list, select = [], lambda n, indices: self.nbr_memory[n][indices]
            sh_imgs,tx_imgs = select('img', idx_sh),select('img', idx_tx)
            #sh1_imgs,sh2_imgs,sh3_imgs = select('img', idx_sh1),select('img', idx_sh2),select('img', idx_sh3)
            #sh4_imgs = select('img',idx_sh4)
            sh_masks, tx_masks = select('mask', idx_sh), select('mask', idx_tx)
            #sh1_masks,sh2_masks,sh3_masks = select('mask',idx_sh1),select('mask',idx_sh2),select('mask',idx_sh3) 
            #sh4_masks = select('mask',idx_sh4)

            # Swap shapes
            with torch.no_grad():
                # we recompute parameters with the current network state
                if self.shared_encoder:
                    sh_features = self.encoder(sh_imgs)
                    local_features = self.local_encoder(sh_imgs)
                    #features_tx = self.transformer_txt(local_features)
                    m = self.predict_meshes(sh_features, local_features, average_txt=average_txt)
                    v,f = m.get_mesh_verts_faces(0)
                    sh_tx = self.predict_textures(f, local_features, average_txt)
                    sh_S = self.predict_scales(sh_features)[:, None]
                    sh_R, sh_T = self.predict_poses(sh_features)
                    sh_bg = self.predict_background(sh_features) if self.pred_background else None
                else:
                    m = self.predict_verts(self.encoder_sh(sh_imgs))
                    local_features = self.local_encoder(sh_imgs)
                    #features_tx = self.transformer_txt(local_features)
                    v,f = m.get_mesh_verts_faces(0)
                    sh_tx = self.predict_textures(f, local_features, average_txt)
                    sh_S = self.predict_scales(self.encoder_sc(sh_imgs))[:, None]
                    sh_R, sh_T = self.predict_poses(self.encoder_pose(sh_imgs))
                    sh_bg = self.predict_background(self.encoder_bg(sh_imgs)) if self.pred_background else None
            sh_mesh = Meshes((verts / scales) * sh_S, faces, sh_tx)
            nbr_list.append([sh_mesh, sh_R, sh_T, sh_bg, sh_imgs, sh_masks])
            
            # Swap textures
            with torch.no_grad():
                # we recompute parameters with the current network state
                if self.shared_encoder:
                    tx_features = self.encoder(tx_imgs)
                    tx_local_features = self.local_encoder(tx_imgs)
                    tx_part_features = self.transformer_part(tx_local_features)
                    m = self.predict_verts(tx_features)
                    v,f = m.get_mesh_verts_faces(0)
                    tx_verts = v + self.predict_disp_verts(v,tx_part_features,tx_features).view(B, -1, 3)
                    tx_S = self.predict_scales(tx_features)[:, None]
                    tx_R, tx_T = self.predict_poses(tx_features)
                    tx_bg = self.predict_background(tx_features) if self.pred_background else None
                else:
                    tx_feat_sc = self.encoder_sc(tx_imgs)
                    tx_feat_sh = self.encoder_sh(tx_imgs)
                    tx_feat_local = self.local_encoder(tx_imgs)
                    #features = self.transformer_part(tx_feat_local)
                    m = self.predict_verts(tx_feat_sh)
                    v,f = m.get_mesh_verts_faces(0)
                    tx_verts = v + self.predict_disp_verts(v,tx_feat_local,tx_feat_sh).view(B, -1, 3)
                    tx_S = self.predict_scales(tx_feat_sc)[:, None]
                    tx_R, tx_T = self.predict_poses(self.encoder_pose(tx_imgs))
                    tx_bg = self.predict_background(self.encoder_bg(tx_imgs)) if self.pred_background else None
            tx_mesh = Meshes(tx_verts * tx_S, faces, textures)
            nbr_list.append([tx_mesh, tx_R, tx_T, tx_bg, tx_imgs, tx_masks])

            loss = 0.
            for nbr_inp in nbr_list:
                nbr_mesh, R, T, bkgs, imgs, masks = nbr_inp
                #imgs_masked = imgs*masks
                rec_sw, alpha_sw = self.layer_renderer(nbr_mesh, R, T).split([3, 1], dim=1)
                rec_sw = rec_sw * alpha_sw + (1 - alpha_sw) * bkgs[:B] if bkgs is not None else rec_sw
                #rec_masked = rec_sw * alpha_sw
                if 'rgb' in losses:
                    loss += self.loss_weights['rgb'] * self.criterion(rec_sw,imgs).flatten(1).mean(1)
                if 'perceptual' in losses:
                    loss += self.loss_weights['perceptual'] * self.perceptual_loss(rec_sw, imgs)
                if 'mask' in losses:
                    loss += self.loss_weights['mask'] * self.mask_loss(alpha_sw, masks)
            losses['neighbor'] = self.loss_weights['neighbor'] * loss
        
        # Pose priors
        if update_pose and 'uniform' in losses:
            losses['uniform'] = self.loss_weights['uniform'] * (self._pose_proba.mean(1) - 1 / K).abs().mean()

        dist = sum(losses.values())
        if K > 1:
            dist, select_idx = dist.view(K, B), self._pose_proba.max(0)[1]
            dist = (self._pose_proba * dist).sum(0)
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = (self._pose_proba * v.view(K, B)).sum(0).mean()

            # For monitoring purpose only
            pose_proba_d = self._pose_proba.detach().cpu()
            self._prob_heads = pose_proba_d.mean(1).tolist()
            self._prob_max = pose_proba_d.max(0)[0].mean().item()
            self._prob_min = pose_proba_d.min(0)[0].mean().item()
            count = torch.zeros(K, B).scatter(0, select_idx[None].cpu(), 1).sum(1)
            self.prop_heads = count / B

        else:
            select_idx = torch.zeros(B).long()
            for k, v in losses.items():
                if v.numel() != 1:
                    losses[k] = v.mean()

        losses['total'] = dist.mean()
        return losses, select_idx
        
    def iter_step(self):
        self.cur_iter += 1
        if self.alternate_optim and self.cur_iter % self.alternate_optim == 0:
            self.pose_step = not self.pose_step

    def step(self):
        self.cur_epoch += 1
        self.direct_field.step()
        self.weights_field.step()
        self.txt_field.step()
        self.deform_field1.step()
        self.deform_field2.step()
        self.deform_field3.step()
        #self.deform_field4.step()
        if self.pred_background:
            self.bkg_generator.step()

    def set_cur_epoch(self, epoch):
        self.cur_epoch = epoch
        self.direct_field.set_cur_milestone(epoch)
        self.weights_field.set_cur_milestone(epoch)
        self.txt_field.set_cur_milestone(epoch)
        self.deform_field1.set_cur_milestone(epoch)
        self.deform_field2.set_cur_milestone(epoch)
        self.deform_field3.set_cur_milestone(epoch)
        #self.deform_field4.set_cur_milestone(epoch)
        if self.pred_background:
            self.bkg_generator.set_cur_milestone(epoch)

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in safe_model_state_dict(state_dict).items():
            if name in state and name != 'T_init':
                try:
                    state[name].copy_(param.data if isinstance(param, nn.Parameter) else param)
                except RuntimeError:
                    print_warning(f'Error load_state_dict param={name}: {list(param.shape)}, {list(state[name].shape)}')
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f'load_state_dict: {unloaded_params} not found')

    ########################
    # Visualization utils
    ########################

    def get_synthetic_textures(self, colored=False):
        subdivide = SubdivideMeshes()
        verts = subdivide(self.mesh_src).verts_packed()
        if colored:
            colors = (verts - verts.min(0)[0]) / (verts.max(0)[0] - verts.min(0)[0])
        else:
            colors = torch.ones(verts.shape, device=verts.device) * 0.8
        return TexturesVertex(verts_features=colors[None])
        
    @torch.no_grad()
    def convert_labels_to_colors(self,labels,filename):
        if labels.shape[-1] != N_K:
            labels = labels.permute(1,2,0)
        labels = labels.unsqueeze(3)
        colors = torch.mul(labels.repeat(1,1,1,3).cpu(),torch.tensor(color_map[:N_K])).sum(dim=2)
        colors = colors.type(torch.uint8).numpy()
        im = Image.fromarray(colors).convert('RGB')
        im.save(filename)
        
    @torch.no_grad()
    def get_outputs_colors(self,colors,k):
        weights = self.weights
        #weights = self.model.generate_part_supervision()
        colors = np.array(colors[:N_K])
        colors = colors.reshape(1,N_K,3)
        colors = np.repeat(colors,weights.shape[1],axis=0)
        colors = torch.from_numpy(colors).to("cuda")
        weight = weights[k]
        weights = (weight==weight.max(dim=1,keepdim=True)[0])
        weights = weights.unsqueeze(2)
        weighted_colors = torch.sum(weights*colors,dim=1)/255.
        #return weighted_colors
        return TexturesVertex(verts_features=weighted_colors[None])

    def get_prototype(self):
        vert = self.mesh_src.get_mesh_verts_faces(0)[0]
        latent_coarse = torch.zeros(1, self.n_features, device=vert.device)
        meshes = self.predict_verts(latent_coarse)
        verts = meshes.get_mesh_verts_faces(0)[0]
        latent_seg = torch.zeros(1, self.n_features, device=verts.device)
        latent = torch.zeros(1,N_K,self.tx_features,device=verts.device)
        disp_verts = []
        disp_verts.append(self.deform_field1(verts,latent[:,0,:]))
        disp_verts.append(self.deform_field2(verts,latent[:,1,:]))
        disp_verts.append(self.deform_field3(verts,latent[:,2,:]))
        #disp_verts.append(self.deform_field4(verts,latent[:,3,:]))
        disp_verts = torch.stack(disp_verts,1) # BKV3
        weights = self.subdivide_weights(latent_seg)
        weights = weights.permute(0,2,1)
        weights = weights.unsqueeze(3).repeat(1,1,1,3)
        disp_verts = (disp_verts*weights).sum(1)
        new_meshes = meshes.offset_verts(disp_verts.view(-1, 3))
        return new_meshes
    
    @use_seed()
    @torch.no_grad()
    def get_random_prototype_views(self, N=10):
        mesh = self.get_prototype()
        if mesh is None:
            return None
            
        mesh.textures = self.get_synthetic_textures(colored=True)
        azim = torch.randint(*self.azim_range, size=(N,))
        elev = torch.randint(*self.elev_range, size=(N,)) if np.diff(self.elev_range)[0] > 0 else self.elev_range[0]
        R, T = look_at_view_transform(dist=self.T_init[-1], elev=elev, azim=azim, device=mesh.device)
        return self.layer_renderer(mesh.extend(N), R, T).split([3, 1], dim=1)[0]

    @torch.no_grad()
    def save_prototype(self, path=None):
        mesh = self.get_prototype()
        if mesh is None:
            return None

        path = path_mkdir(path or Path('.'))
        d, elev = self.T_init[-1], np.mean(self.elev_range)
        mesh.textures = self.get_synthetic_textures()
        save_mesh_as_obj(mesh, path / 'proto.obj')
        save_mesh_as_gif(mesh, path / 'proto_li.gif', dist=d, elev=elev, renderer=self.layer_renderer, eye_light=True)
        mesh.textures = self.get_synthetic_textures(colored=True)
        save_mesh_as_gif(mesh, path / 'proto_uv.gif', dist=d, elev=elev, renderer=self.layer_renderer)

    ########################
    # Evaluation utils
    ########################

    @torch.no_grad()
    def quantitative_eval(self, loader, device, evaluator=None):
        self.eval()
        if loader.dataset.name in ['cub_200']:
            if evaluator is None:
                evaluator = ProxyEvaluator()
            for inp, _ in loader:
                mask_gt = inp['masks'].to(device)
                meshes, meshes_part, (R, T), bkgs = self.predict_mesh_pose_bkg(inp['imgs'].to(device))
                mask_pred = self.layer_renderer(meshes, R, T, viz_purpose=True).split([3, 1], dim=1)[1]  # (K*B)CHW
                if mask_pred.shape != mask_gt.shape:
                    mask_pred = F.interpolate(mask_pred, mask_gt.shape[-2:], mode='bilinear', align_corners=False)
                mask_pred = (mask_pred > 0.5).float()
                evaluator.update(mask_pred, mask_gt)
        else:
            if loader.dataset.name == 'p3d_car':
                print_warning('make sure that the canonical axes of predicted shapes correspond to the GT shapes axes')
            if evaluator is not None:
                evaluator = PartEvaluator()
                for inp, labels in loader:
                    if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                        break
                    meshes, meshes_part, (R, T), bkgs = self.predict_mesh_pose_bkg(inp["imgs"].to(device))
                    if not torch.all(inp["poses"] == -1):
                        verts,faces = meshes_part.verts_padded(), meshes_part.faces_padded()
                        R_gt,T_gt = map(lambda t: t.squeeze(2), inp["poses"].to(device).split([3,1],dim=2))
                        verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
                    meshes_part = PartMesh(verts=verts, faces=faces, labels=meshes_part.vert_labels)
                    evaluator.update(meshes_part, torch_to(labels, device))
            else:
                evaluator = PartEvaluator()
                for inp, labels in loader:
                    if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                        break
                    meshes, meshes_part, (R, T), bkgs = self.predict_mesh_pose_bkg(inp["imgs"].to(device))
                    if not torch.all(inp["poses"] == -1):
                        verts,faces = meshes_part.verts_padded(), meshes_part.faces_padded()
                        R_gt,T_gt = map(lambda t: t.squeeze(2), inp["poses"].to(device).split([3,1],dim=2))
                        verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
                    meshes_part = PartMesh(verts=verts, faces=faces, labels=meshes_part.vert_labels)
                    evaluator.update(meshes_part, torch_to(labels, device))
                """
                evaluator = MeshEvaluator()
                for inp, labels in loader:
                    if isinstance(labels, torch.Tensor) and torch.all(labels == -1):
                        break
    
                    meshes, meshes_part, (R, T), bkgs = self.predict_mesh_pose_bkg(inp['imgs'].to(device))
                    if not torch.all(inp['poses'] == -1):
                        # we use x_pred @ R_pred + T_pred = x_gt @ R_gt + T_gt to align predicted mesh with GT mesh
                        verts, faces = meshes.verts_padded(), meshes.faces_padded()
                        R_gt, T_gt = map(lambda t: t.squeeze(2), inp['poses'].to(device).split([3, 1], dim=2))
                        verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
                        meshes = Meshes(verts=verts, faces=faces, textures=meshes.textures)
                    evaluator.update(meshes, torch_to(labels, device))
                """
        return OrderedDict(zip(evaluator.metrics.names, evaluator.metrics.values))
        
    @torch.no_grad()
    def qualitative_eval(self, loader, device, path=None, N=32):
        path = path or Path('.')
        self.eval()
        self.save_prototype(path / 'model')

        n_zeros, NI = int(np.log10(N - 1)) + 1, max(N // loader.batch_size, 1)
        for j, (inp, _) in enumerate(loader):
            if j == NI:
                break
            imgs = inp['imgs'].to(device)
            masks = inp['masks'].to(device)
            meshes, part_meshes, (R, T), bkgs = self.predict_mesh_pose_bkg(imgs)
            rec, alpha = self.layer_renderer(meshes, R, T).split([3, 1], dim=1)  # (K*B)CHW
            part_rec, part_alpha = self.part_renderer(part_meshes, R, T).split([N_K, 1], dim=1)
            if bkgs is not None:
                rec = rec * alpha + (1 - alpha) * bkgs
            B, NV = len(imgs), 50
            d, e = self.T_init[-1], np.mean(self.elev_range)
            for k in range(B):
                i = str(j*B+k).zfill(n_zeros)
                convert_to_img(imgs[k]).save(path / f'{i}_inpraw.png')
                convert_to_img(rec[k]).save(path / f'{i}_inprec_full.png')
                convert_to_img(masks[k]).save(path / f'{i}_masks.png')
                convert_to_img(alpha[k]).save(path / f'{i}_masks_rec.png')
                self.convert_labels_to_colors(part_rec[k],os.path.join(path,f'{i}_part_rec.png'))
                #self.convert_labels_to_colors(part_fore[k],os.path.join(path,f'{i}_part.png'))
                #self.convert_labels_to_colors(part_alpha[k],os.path.join(path,f'{i}_part_alpha.png'))
                #self.convert_labels_to_colors(masks[k],os.path.join(path,f'{i}_masks.png'))
                if self.pred_background:
                    convert_to_img(bkgs[k]).save(path / f'{i}_inprec_wbkg.png')

                mcenter = normalize(meshes[k])
                save_mesh_as_gif(mcenter, path / f'{i}_meshabs.gif', n_views=NV, dist=d, elev=e, renderer=self.layer_renderer)
                save_mesh_as_obj(mcenter, path / f'{i}_mesh.obj')
                mcenter_part = normalize(part_meshes[k])
                mcenter_part.textures = self.get_outputs_colors(color_map,k)
                save_mesh_as_gif(mcenter_part, path / f'{i}_meshuv_weights.gif', dist=d, elev=e, renderer=self.layer_renderer)