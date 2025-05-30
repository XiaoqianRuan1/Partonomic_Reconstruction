from collections import defaultdict, OrderedDict
import pandas as pd
from pathlib import Path
from pytorch3d.ops import sample_points_from_meshes as sample_points, iterative_closest_point as torch_icp
import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments
from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from .chamfer import chamfer_distance, part_chamfer_distance
from .logger import print_log
from .mesh import normalize
from math import log10
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.structures.utils import list_to_packed, list_to_padded, padded_to_list
import logging

EPS = 1e-7
CHAMFER_FACTOR = 10  # standard multiplicative factor to report Chamfer, see OccNet or DVR


class PartMesh(Meshes):
    def __init__(self,verts,faces,labels,textures=None):
        """
        add a new attribute, labels
        """
        super().__init__(verts,faces)
        self.vert_labels = labels # similar with verts_features_padded
        self.verts = verts
        self.faces = faces
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



def PSNR(mse, peak=1.0):
    return 10*log10((peak**2)/mse)
    
class SegMetric:
    def __init__(self, values=0.):
        assert isinstance(values, dict)
        self.miou = values.miou
        self.oa = values.get('oa', None) 
        self.miou = values.miou
        self.miou = values.miou

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, N=1):
        if isinstance(val, torch.Tensor):
            assert val.numel() == 1
            val = val.item()
        self.val = val
        self.sum += val * N
        self.count += N
        self.avg = self.sum / self.count if self.count != 0 else 0


class Metrics:
    log_data = True

    def __init__(self, *names, log_file=None, append=False):
        self.names = list(names)
        self.meters = defaultdict(AverageMeter)
        if log_file is not None and self.log_data:
            self.log_file = Path(log_file)
            if not self.log_file.exists() or not append:
                with open(self.log_file, mode='w') as f:
                    f.write("iteration\tepoch\tbatch\t" + "\t".join(self.names) + "\n")
        else:
            self.log_file = None

    def log_and_reset(self, *names, it=None, epoch=None, batch=None):
        self.log(it, epoch, batch)
        self.reset(*names)

    def log(self, it, epoch, batch):
        if self.log_file is not None:
            with open(self.log_file, mode="a") as file:
                file.write(f"{it}\t{epoch}\t{batch}\t" + "\t".join(map("{:.6f}".format, self.values)) + "\n")

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self[name].reset()

    def read_log(self):
        if self.log_file is not None:
            return pd.read_csv(self.log_file, sep='\t', index_col=0)
        else:
            return pd.DataFrame()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ', '.join(['{}={:.4f}'.format(name, self[name].avg) for name in self.names])

    def __len__(self):
        return len(self.names)

    @property
    def values(self):
        return [self[name].avg for name in self.names]

    def update(self, *name_val, N=1):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v, N=N)
        else:
            assert len(name_val) == 2
            name, val = name_val
            if name not in self.names:
                raise KeyError(f'{name} not in current metrics')
            if isinstance(val, (tuple, list)):
                self[name].update(val[0], N=val[1])
            else:
                self[name].update(val, N=N)

    def get_named_values(self, filter_fn=None):
        names, values = self.names, self.values
        if filter_fn is not None:
            zip_fn = lambda k_v: filter_fn(k_v[0])
            names, values = map(list, zip(*filter(zip_fn, zip(names, values))))
        return list(zip(names, values))


class MeshEvaluator:
    """
    Mesh evaluation class by computing similarity metrics between predicted mesh and GT.
    Code inspired from https://github.com/autonomousvision/differentiable_volumetric_rendering (see im2mesh/eval.py)
    """
    default_names = ['chamfer-L1', 'chamfer-L1-ICP', 'normal-cos', 'normal-cos-ICP']

    def __init__(self, names=None, log_file=None, run_icp=True, estimate_scale=True, anisotropic_scale=True,
                 icp_type='gradient', fast_cpu=False, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)
        self.run_icp = run_icp
        self.estimate_scale = estimate_scale
        self.ani_scale = anisotropic_scale
        self.icp_type = icp_type
        assert icp_type in ['normal', 'gradient']
        self.fast_cpu = fast_cpu
        self.N = 50000 if fast_cpu else 100000
        print_log('MeshEvaluator init: run_icp={}, estimate_scale={}, anisotropic_scale={}, icp_type={}, n_iter={}'
                  .format(run_icp, estimate_scale, anisotropic_scale, icp_type, self.n_iter))

    @property
    def n_iter(self):
        if self.icp_type == 'normal':
            return 10 if self.fast_cpu else 30
        else:
            return 30 if self.fast_cpu else 100

    def update(self, mesh_pred, labels):
        pc_gt, norm_gt = labels['points'], labels['normals']
        vox_gt = labels.get('voxels')
        res = self.evaluate(mesh_pred, pc_gt=pc_gt, norm_gt=norm_gt, vox_gt=vox_gt)
        self.metrics.update(res, N=len(mesh_pred))
    
    def results_output(self, mesh_preds, labels):
        f = open("/data/unicorn-main/runs/shapenet_test/1014_table_test/results.txt",'a')
        pc_gts, norm_gts = labels['points'], labels['normals']
        vox_gt = labels.get('voxels')
        for i in range(len(mesh_preds)):
            mesh_pred = mesh_preds[i]
            pc_gt, norm_gt = pc_gts[i].unsqueeze(0),norm_gts[i].unsqueeze(0)
            assert abs(pc_gt.abs().max() - 0.5) < 0.01  # XXX GT should fit in the unit cube [-0.5, 0.5]^3
            pc_pred, norm_pred = sample_points(mesh_pred, self.N, return_normals=True)
            pc_pred = pc_pred[0].unsqueeze(0)
            if self.N < len(pc_gt):
                idxs = torch.randperm(len(pc_gt))[:self.N]
                pc_gt, norm_gt = pc_gt[:, idxs], norm_gt[:, idxs]
    
            use_scale, ani_scale, n_iter = self.estimate_scale, self.ani_scale, self.n_iter
            results = []
            if self.run_icp:
                # Normalize mesh to be centered around 0 and fit inside the unit cube for better ICP
                mesh_pred = normalize(mesh_pred)
                pc_pred2, norm_pred2 = sample_points(mesh_pred, self.N, return_normals=True)
                if self.icp_type == 'normal':
                    pc_pred_icp, RTs = torch_icp(pc_pred2, pc_gt, estimate_scale=use_scale, max_iterations=n_iter)[2:4]
                else:
                    from .icp import gradient_icp
                    pc_pred2 = pc_pred2[0].unsqueeze(0)
                    pc_pred_icp, RTs = gradient_icp(pc_pred2, pc_gt, use_scale, ani_scale, lr=0.01, n_iter=n_iter)
                pc_preds, norm_preds, tags = [pc_pred, pc_pred_icp], [norm_pred, norm_pred2], ['', '-ICP']
                #pc_preds, norm_preds, tags = [pc_pred_icp], [norm_pred2], ['-ICP']
            else:
                pc_preds, norm_preds, tags = [pc_pred], [norm_pred], ['']
    
            for pc, norm, tag in zip(pc_preds, norm_preds, tags):
                chamfer_L1, normal = chamfer_distance(pc_gt, pc, x_normals=norm_gt, y_normals=norm,
                                                          return_L1=True, return_mean=True)
                chamfer_L1 = chamfer_L1 * CHAMFER_FACTOR
                results += [('chamfer-L1' + tag, chamfer_L1.item()), ('normal-cos' + tag, 1 - normal.item())]
            results = OrderedDict(list(filter(lambda x: x[0] in self.names, results)))
            f.writelines(str(results['chamfer-L1-ICP']))
            f.writelines("\n")

    def evaluate(self, mesh_pred, pc_gt, norm_gt, vox_gt=None):
        assert abs(pc_gt.abs().max() - 0.5) < 0.01  # XXX GT should fit in the unit cube [-0.5, 0.5]^3
        pc_pred, norm_pred = sample_points(mesh_pred, self.N, return_normals=True)
        if self.N < len(pc_gt):
            idxs = torch.randperm(len(pc_gt))[:self.N]
            pc_gt, norm_gt = pc_gt[:, idxs], norm_gt[:, idxs]

        use_scale, ani_scale, n_iter = self.estimate_scale, self.ani_scale, self.n_iter
        results = []
        if self.run_icp:
            # Normalize mesh to be centered around 0 and fit inside the unit cube for better ICP
            mesh_pred = normalize(mesh_pred)
            pc_pred2, norm_pred2 = sample_points(mesh_pred, self.N, return_normals=True)
            if self.icp_type == 'normal':
                pc_pred_icp, RTs = torch_icp(pc_pred2, pc_gt, estimate_scale=use_scale, max_iterations=n_iter)[2:4]
            else:
                from .icp import gradient_icp
                pc_pred_icp, RTs = gradient_icp(pc_pred2, pc_gt, use_scale, ani_scale, lr=0.01, n_iter=n_iter)
            pc_preds, norm_preds, tags = [pc_pred, pc_pred_icp], [norm_pred, norm_pred2], ['', '-ICP']
        else:
            pc_preds, norm_preds, tags = [pc_pred], [norm_pred], ['']

        for pc, norm, tag in zip(pc_preds, norm_preds, tags):
            chamfer_L1, normal = chamfer_distance(pc_gt, pc, x_normals=norm_gt, y_normals=norm,
                                                      return_L1=True, return_mean=True)
            chamfer_L1 = chamfer_L1 * CHAMFER_FACTOR
            results += [('chamfer-L1' + tag, chamfer_L1.item()), ('normal-cos' + tag, 1 - normal.item())]
        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()


class PartEvaluator:
    default_names = ["chamfer-L1", "chamfer-L1-ICP", "part-chamfer", "part-chamfer-ICP"]
    def __init__(self, names=None, log_file=None, run_icp=True, estimate_scale=True, anisotropic_scale=True,
                 icp_type='gradient', fast_cpu=False, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)
        self.run_icp = run_icp
        self.estimate_scale = estimate_scale
        self.ani_scale = anisotropic_scale
        self.icp_type = icp_type
        self.fast_cpu = fast_cpu
        self.N = 50000 if fast_cpu else 100000
        print_log('PartEvaluator init: run_icp={}, estimate_scale={}, anisotropic_scale={}, icp_type={}, n_iter={}'
                  .format(run_icp, estimate_scale, anisotropic_scale, icp_type, self.n_iter))      
    
    @property
    def n_iter(self):
        if self.icp_type == "normal":
            return 10 if self.fast_cpu else 30
        else:
            return 30 if self.fast_cpu else 100
    
    def update(self, mesh_pred, labels):
        pc_gt, part_gt = labels["points"], labels["labels"]
        vox_gt = labels.get("voxels")
        res = self.evaluate(mesh_pred, pc_gt=pc_gt, part_gt=part_gt, vox_gt=vox_gt)
        self.metrics.update(res, N=len(mesh_pred))
        
    def evaluate(self, mesh_pred, pc_gt, part_gt, vox_gt=None):
        """
        mesh_pred: the predicted part mesh
        pc_gt: point cloud ground
        part_gt: part labels
        Evaluation: As for the same part for the prediction and ground truth, we calculate their chamfer distance.
        """ 
        assert abs(pc_gt.abs().max() - 0.5) < 0.01 # all GT should fit in the unit cube [-0.5, 0.5]
        pc_pred, label_pred = sample_points_from_part_meshes(mesh_pred, self.N, return_labels=True)
        if self.N < len(pc_gt):
            idxs = torch.randperm(len(pc_gt))[:self.N]
            pc_gt = pc_gt[:, idxs]
        use_scale, ani_scale, n_iter = self.estimate_scale, self.ani_scale, self.n_iter
        results = []
        if self.run_icp:
            mesh_pred = normalize(mesh_pred)
            pc_pred2, label_pred2 = sample_points_from_part_meshes(mesh_pred, self.N, return_labels=True)
            if self.icp_type == "normal":
                pc_pred_icp, RTs = torch_icp(pc_pred2, pc_gt, estimate_scale=use_scale, max_iterations=n_iter)[2:4]
            else:
                from .icp import gradient_icp
                pc_pred_icp, RTs = gradient_icp(pc_pred2, pc_gt, use_scale, ani_scale, lr=0.01, n_iter=n_iter)   
            pc_preds, label_preds, tags = [pc_pred, pc_pred_icp], [label_pred, label_pred2], ["", "-ICP"]
        else:
           pc_preds, tags = [pc_pred], [""]
        for pc, label, tag in zip(pc_preds, label_preds, tags):  
            chamfer_L1,_ = chamfer_distance(pc_gt, pc, return_L1=True, return_mean=True)
            part_chamfer_L1 = part_chamfer_distance(pc_gt, pc, part_gt, label, return_L1=True, return_mean=True) 
            part_chamfer_L1 = part_chamfer_L1 * CHAMFER_FACTOR
            chamfer_L1 = chamfer_L1 * CHAMFER_FACTOR
            results += [('chamfer-L1' + tag, chamfer_L1.item()),('part-chamfer' + tag, part_chamfer_L1.item())]
        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)         
   
    def results_output(self, mesh_preds, labels, txt_file):
        f = open(txt_file,'a')
        pc_gts, part_gts = labels["points"].to(mesh_preds.device), labels["labels"].to(mesh_preds.device)
        vox_gt = labels.get("voxels")
        for i in range(len(mesh_preds)):
            mesh_part = mesh_preds[i]
            mesh_pred = PartMesh(verts=mesh_part.verts, faces=mesh_part.faces, labels=mesh_part.vert_labels[i].unsqueeze(0))
            pc_gt, part_gt = pc_gts[i].unsqueeze(0),part_gts[i].unsqueeze(0)
            assert abs(pc_gt.abs().max() - 0.5) < 0.01  # XXX GT should fit in the unit cube [-0.5, 0.5]^3
            pc_pred, label_pred = sample_points_from_part_meshes(mesh_pred, self.N, return_labels=True)
            pc_pred = pc_pred[0].unsqueeze(0)
            if self.N < len(pc_gt):
                idxs = torch.randperm(len(pc_gt))[:self.N]
                pc_gt = pc_gt[:, idxs]
            use_scale, ani_scale, n_iter = self.estimate_scale, self.ani_scale, self.n_iter
            results = []
            if self.run_icp:
                # Normalize mesh to be centered around 0 and fit inside the unit cube for better ICP
                mesh_pred = normalize(mesh_pred)
                pc_pred2, label_pred2 = sample_points_from_part_meshes(mesh_pred, self.N, return_labels=True)
                if self.icp_type == "normal":
                    pc_pred_icp, RTs = torch_icp(pc_pred2, pc_gt, estimate_scale=use_scale, max_iterations=n_iter)[2:4]
                else:
                    from .icp import gradient_icp
                    pc_pred2 = pc_pred2[0].unsqueeze(0)
                    pc_pred_icp, RTs = gradient_icp(pc_pred2, pc_gt, use_scale, ani_scale, lr=0.01, n_iter=n_iter)   
                pc_preds, label_preds, tags = [pc_pred, pc_pred_icp], [label_pred, label_pred2], ["", "-ICP"]
            else:
                pc_preds, norm_preds, tags = [pc_pred], [norm_pred], ['']
            for pc, label, tag in zip(pc_preds, label_preds, tags):  
                chamfer_L1,_ = chamfer_distance(pc_gt, pc, return_L1=True, return_mean=True)
                part_chamfer_L1 = part_chamfer_distance(pc_gt, pc, part_gt, label, return_L1=True, return_mean=True) 
                part_chamfer_L1 = part_chamfer_L1 * CHAMFER_FACTOR
                chamfer_L1 = chamfer_L1 * CHAMFER_FACTOR
                results += [('chamfer-L1' + tag, chamfer_L1.item()),('part-chamfer' + tag, part_chamfer_L1.item())]
            f.writelines(str(results))
            f.writelines("\n")
   
    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()
                
    
class ProxyEvaluator:
    default_names = ['mask_iou']

    def __init__(self, names=None, log_file=None, append=False):
        self.names = names if names is not None else self.default_names
        self.metrics = Metrics(*self.names, log_file=log_file, append=append)

    def update(self, mask_pred, mask_gt):
        for k in range(len(mask_pred)):
            self.metrics.update(self.evaluate(mask_pred[k], mask_gt[k]))

    def result_output(self, mask_pred, mask_gt, txt_path):
        with open(txt_path,"a") as f:
            for i in range(mask_pred.shape[0]):
                miou = (mask_pred[i] * mask_gt[i]).sum() / (mask_pred[i] + mask_gt[i]).clamp(0, 1).sum()
                f.writelines(str(miou))
                f.writelines("\n")

    def evaluate(self, mask_pred, mask_gt):
        results = []
        miou = (mask_pred * mask_gt).sum() / (mask_pred + mask_gt).clamp(0, 1).sum()
        results += [('mask_iou', miou.item())]
        results = list(filter(lambda x: x[0] in self.names, results))
        return OrderedDict(results)

    def compute(self):
        return self.metrics.values

    def __repr__(self):
        return self.metrics.__repr__()

    def log_and_reset(self, it, epoch, batch):
        self.metrics.log_and_reset(it=it, epoch=epoch, batch=batch)

    def read_log(self):
        return self.metrics.read_log()


def sample_points_from_part_meshes(meshes, num_samples, return_labels=False):
    """
    an advanced version of pytorch3d.ops.sample_points_from_meshes by changing textures as vertex_labels.
    meshes: PartMesh
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")
    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")
    if return_labels and meshes.vert_labels is None:
        raise ValueError("Meshes do not contain labels.")
    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid) # Non empty meshes.
    #Initalize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(verts, faces)
        max_faces = meshes.num_faces_per_mesh().max().item()
        areas_padded = packed_to_padded(areas, mesh_to_face[meshes.valid], max_faces)
        sample_face_idxs = areas_padded.multinomial(num_samples, replacement=True)
        sample_face_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes,1)
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:,0], face_verts[:,1], face_verts[:,2]
    w0, w1, w2 = _rand_barycentric_coords(num_valid_meshes, num_samples, verts.dtype, verts.device)
    a = v0[sample_face_idxs]  # (N, num_samples, 3)
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c
    if return_labels:
        pix_to_face = sample_face_idxs.view(len(meshes),num_samples,1,1)
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)
        dummy = torch.zeros((len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32)
        fragments = MeshFragments(pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy)
        faces = meshes.faces_packed()
        verts_features_packed = meshes.verts_labels_packed() 
        faces_verts_features = verts_features_packed[faces]
        labels = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, faces_verts_features)
        labels = labels[:,:,0,0,:]
        if return_labels:
            return samples, labels
        return samples
