from copy import deepcopy
from PIL import Image
import yaml
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms.functional import to_tensor
from utils import path_exists
import os

#DATASETS_PATH = "/data/unicorn-main1/datasets/shapenet_nmr/"
#part_root = "/data/unicorn-part/data/ShapeNetPart/new1"
#part_root = "/mnt/sde1/xiaoqianruan/unicorn-main_2d_ground_truth/data/ShapeNetPart/ground/"
data_path = "/data/unicorn-part/data/PartNet/"

N_K = 4
color_map = [
    [255,0,0],
    [0,0,255],
    [0,255,0],
    [128,0,128],
    [255,255,255],
    #[0,0,0],
]

class TestPart(TorchDataset):
    root = data_path
    name = 'partnet'
    n_channels = 3
    
    def __init__(self, split, img_size, categories=None, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.data_path = data_path
        assert len(kwargs) == 0
        try:
            self.data_path = path_exists(data_path)
        except FileNotFoundError:
            self.data_path = path_exists(TMP_PATH / 'datasets')
        
        with open(os.path.join(self.data_path,'metadata.yaml')) as fp:
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
        category_path = os.path.join(self.data_path,"category")
        for c in categories:
            with open(os.path.join(category_path,c,f'{split}.txt'), "r") as f:
                names = f.read().split('\n')
            names = list(filter(lambda x: len(x) > 0, names))
            models += [{'category': c, 'split':split, 'model': n} for n in names]
        return models

    def __len__(self):
        return self.n_models

    def __getitem__(self, idx):
        cat = self.models[idx]['category']
        model = self.models[idx]['model']
        split = self.models[idx]['split']
        image_path = os.path.join(self.data_path, "images", str(cat), str(model))
        cameras = np.load(os.path.join(self.data_path,"cameras.npz"))
        pc_npz,label_npz = self.read_ground_truth(os.path.join(self.data_path,"ground_truth",str(cat),str(split),str(model)))
        points = torch.Tensor(pc_npz)
        labels = torch.Tensor(label_npz)
        imgs = to_tensor(Image.open(os.path.join(image_path,"0000.png")))
        poses = self.adjust_extrinsics(torch.Tensor(cameras[f'world_mat_0']))
        return ({'imgs': imgs,'poses': poses},{'points': points,'labels': labels})

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
    
    def read_ground_truth(self,data_path):
        split = data_path.split("/")[-2]
        if split == "train":
            seg = np.zeros((10000)).astype(np.int64)
            one_hot_seg = self.convert_to_onehot(seg,N_K)
            normalized_points = np.zeros((10000,3))
        else:
            rgb_file = os.path.join(data_path,"point_cloud/sample-points-all-pts-nor-rgba-10000.txt")
            part_file = os.path.join(data_path,"point_cloud/sample-points-all-label-10000.txt")
            rgb_data = np.loadtxt(rgb_file)
            part_data = np.loadtxt(part_file)
            points, rgb = rgb_data[:,0:3], rgb_data[:,6:9]/255.0
            normalized_points = self.normalize_point_cloud(points)
            seg = part_data.astype(np.int64)
            one_hot_seg = self.convert_to_onehot(seg,N_K)
        return normalized_points,one_hot_seg
        
    def normalize_point_cloud(self,points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        max_val = np.max(np.abs(centered_points))
        normalized_points = centered_points / (2 * max_val)
        return normalized_points