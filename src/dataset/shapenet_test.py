from copy import deepcopy
from PIL import Image
import yaml
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms.functional import to_tensor
from utils import path_exists
import os

DATASETS_PATH = "/data/unicorn-main1/datasets/shapenet_nmr/"
part_root = "/data/unicorn-part/data/ShapeNetPart/new1/"
part_ground = "/data/unicorn-part/data/ShapeNetPart/ground/" # the path for ShapeNet-Part

color_map = [
    [255,0,0],
    [0,0,255],
    [0,255,0],
    [200,0,255],
    [255,255,255],
    #[0,0,0],
]

N_K = 4

class ShapeNetTest(TorchDataset):
    root = DATASETS_PATH
    name = 'shapenet_nmr'
    img_size = (64, 64)
    n_channels = 3
    n_tot_views = 24
    n_views = 1
    
    def __init__(self, split, n_views=1, categories=None, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.n_views = n_views
        self.flatten_views = kwargs.pop('flatten_views',True)
        self.include_test = kwargs.pop('include_test',False)
        assert len(kwargs) == 0
        try:
            self.data_path = path_exists(os.path.join(DATASETS_PATH,self.name))
        except FileNotFoundError:
            self.data_path = path_exists(TMP_PATH / 'datasets' / self.name)
        
        with open(self.data_path / 'metadata.yaml') as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        indices = list(cfg.keys())
        cat2idx = {n: k for k in cfg for n in cfg[k]['name'].split(',')}
        if categories is None:
            categories = indices
        else:
            categories = [categories] if isinstance(categories, str) else categories
            categories = list({cat2idx[c] for c in categories})

        self.models = self.get_models(self.split, categories)
        if self.include_test and self.split == 'train':
            self.models += self.get_models('val', categories) + self.get_models('test', categories)
        self.n_models = len(self.models)
        self.part_path = part_root

        self._R_col_adj = torch.Tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self._R_row_adj = torch.Tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        self._pc_adj = torch.Tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    def get_models(self, split, categories):
        models = []
        for c in categories:
            with open(self.data_path / c / f'new_{split}.lst', 'r') as f:
                names = f.read().split('\n')
            names = list(filter(lambda x: len(x) > 0, names))
            models += [{'category': c, 'model': n} for n in names]
        return models

    @property
    def is_sv_train(self):
        return self.split == 'train' and self.n_views == 1

    def __len__(self):
        if self.split == 'val':
            return 32  # XXX we use only 32 instances for fast validation
        elif self.is_sv_train and self.flatten_views:
            return self.n_models * self.n_tot_views 
        else:
            return self.n_models

    def __getitem__(self, idx):
        if self.is_sv_train and self.flatten_views:
            # XXX we consider each view as independent samples
            idx, indices = idx % self.n_models, [idx // self.n_models]
        else:
            indices = range(self.n_tot_views)
            if self.n_views < self.n_tot_views:
                indices = np.random.choice(indices, self.n_views, replace=False)

        cat = self.models[idx]['category']
        model = self.models[idx]['model']
        path = self.data_path / cat / model
        part_path = os.path.join(self.part_path, str(cat)+"/"+str(model))
        cameras = np.load(path / 'cameras.npz')
        ground_path = os.path.join(part_ground, cat, model)
        pc_npz = np.load(os.path.join(ground_path, 'pointcloud.npz'))
        points = torch.Tensor(pc_npz['points']) @ self._pc_adj
        normals = torch.Tensor(pc_npz['normals']) @ self._pc_adj
        labels = torch.Tensor(self.convert_to_onehot(pc_npz['labels'],N_K))
         
        
        imgs, masks, poses = [], [], []
        parts = []
        for i in indices:
            imgs.append(to_tensor(Image.open(path / 'image' / '{}.png'.format(str(i).zfill(4)))))
            masks.append(to_tensor(Image.open(path / 'mask' / '{}.png'.format(str(i).zfill(4))).convert('L')))
            poses.append(self.adjust_extrinsics(torch.Tensor(cameras[f'world_mat_{i}'])))  # 3x4
            parts.append(self.generate_labels(os.path.join(part_path,'{}.png'.format(str(i).zfill(4)))))
            #parts.append(to_tensor(Image.open(os.path.join(part_path,'{}.png'.format(str(i).zfill(4))))))
            
        if self.n_views > 1:
            return ({'imgs': torch.stack(imgs), 'masks': torch.stack(masks), 'poses': torch.stack(poses), 'parts': torch.stack(parts)},
                    {'points': points, 'normals': normals, 'labels': labels})
        else:
            return ({'imgs': imgs[0], 'masks': masks[0], 'poses': poses[0], 'parts': parts[0]},
                    {'points': points, 'normals': normals, 'labels': labels})

    def adjust_extrinsics(self, P):
        R, T = torch.split(P[:-1], [3, 1], dim=1)
        R = self._R_row_adj @ R.T @ self._R_col_adj
        return torch.cat([R, T], dim=1)
        
    def generate_labels(self,image_path):
        labels = self.assign_pixels_to_cluster(image_path)
        num_classes = len(color_map)
        one_hot_labels = self.convert_to_onehot(labels.flatten(),num_classes)
        one_hot_labels = one_hot_labels.reshape(labels.shape[0],labels.shape[1],num_classes)
        return one_hot_labels
    
    def convert_to_onehot(self,labels,num_classes):
        one_hot = np.zeros((labels.size,num_classes))
        one_hot[np.arange(labels.size),labels] = 1
        return one_hot
    
    def assign_pixels_to_cluster(self,image_path):
        image = Image.open(image_path)
        pixels = np.array(image).reshape(-1,3).astype(np.float32)
        centers = np.array(color_map,dtype=np.float32)
        distances = np.linalg.norm(pixels[:,np.newaxis]-centers,axis=2)
        labels = np.argmin(distances,axis=1)
        return labels.reshape(image.size[1],image.size[0])