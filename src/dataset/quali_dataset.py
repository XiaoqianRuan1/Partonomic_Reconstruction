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
part_ground = "/data/unicorn-part/data/ShapeNetPart/ground/"

N_K = 4

class TestDataset(TorchDataset):
    root = DATASETS_PATH
    name = 'shapenet_nmr'
    n_channels = 3
    
    def __init__(self, split, img_size, categories=None, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
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
        self.n_models = len(self.models)

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

    def __len__(self):
        return self.n_models

    def __getitem__(self, idx):
        cat = self.models[idx]['category']
        model = self.models[idx]['model']
        path = self.data_path / cat / model
        cameras = np.load(path / 'cameras.npz')
        ground_path = os.path.join(part_ground, cat, model)
        pc_npz = np.load(os.path.join(ground_path, 'pointcloud.npz'))
        points = torch.Tensor(pc_npz['points']) @ self._pc_adj
        normals = torch.Tensor(pc_npz['normals']) @ self._pc_adj
        labels = torch.Tensor(self.convert_to_onehot(pc_npz['labels'],N_K))
        imgs = to_tensor(Image.open(path / 'image' / '0000.png'))
        poses = self.adjust_extrinsics(torch.Tensor(cameras['world_mat_0']))
        return ({'imgs': imgs, 'poses': poses},{'points': points,'labels': labels})

    def adjust_extrinsics(self, P):
        R, T = torch.split(P[:-1], [3, 1], dim=1)
        R = self._R_row_adj @ R.T @ self._R_col_adj
        return torch.cat([R, T], dim=1)
    
    def convert_to_onehot(self,labels,num_classes):
        one_hot = np.zeros((labels.size,num_classes))
        one_hot[np.arange(labels.size),labels] = 1
        return one_hot