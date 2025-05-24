from copy import deepcopy
from functools import lru_cache
from PIL import Image

import numpy as np
from random import random
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import ToTensor, Compose, Resize, functional as Fvision, InterpolationMode
from scipy.io import loadmat

from utils import path_exists, use_seed
from utils.image import square_bbox
from utils.path import DATASETS_PATH, TMP_PATH
import os
from torchvision.transforms.functional import to_tensor


PADDING_BBOX = 0.05
JITTER_BBOX = 0.05
RANDOM_FLIP = True
RANDOM_JITTER = True
EVAL_IMG_SIZE = (256, 256)

KP_LEFT_RIGHT_PERMUTATION = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1
KP_NAMES = ['Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing',
            'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat']
KP_COLOR_MAPPING = [
    (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255),
    (255, 255, 0), (0, 0, 255), (0, 128, 255), (128, 0, 255), (0, 128, 0),
    (128, 0, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
]

N_K = 6

color_map = [[0,0,255],
              [0,255,0],
              [255,0,0],
              [200,0,255],
              [0,255,255],
              [255,255,0],
              [0,0,0]]
              
mask_map = [[255,255,255],
              [255,255,255],
              [255,255,255],
              [255,255,255],
              [255,255,255],
              [255,255,255],
              [0,0,0]]              

part_path = "/data/unicorn-part/data/cub_new/"
DATASETS_PATH = "/data/unicorn-main1/datasets"

class CUBTest(TorchDataset):
    root = DATASETS_PATH
    name = 'cub_200'
    n_channels = 3
    n_views = 1

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        try:
            self.data_path = path_exists(os.path.join(DATASETS_PATH,self.name,'new_images'))
        except FileNotFoundError:
            self.data_path = path_exists(TMP_PATH / 'datasets' / self.name / 'new_images')
        #path = self.data_path.parent / 'cachedir' / 'cub' / 'data' / f'{split}_cub_cleaned.mat'
        #path = self.data_path.parent / f'{split}.lst'
        #self.data = loadmat(path, struct_as_record=False, squeeze_me=True)['images']
        self.data = self.get_models()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP) and self.split == 'train'
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER) and self.split == 'train'
        self.eval_mode = kwargs.pop('eval_mode', False)
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        part_data = os.path.join(part_path,data)
        if os.path.isdir(part_data):  # Double-check it’s not a directory
            raise ValueError(f"Path {part_data} is a directory, not a file.")
        part_img = Image.open(part_data).resize((self.img_size[0],self.img_size[0]))
        img = Image.open(self.data_path / data).resize((self.img_size[0],self.img_size[0])).convert('RGB')
        # Horizontal flip
        part = self.generate_labels(part_img)
        mask = self.convert_labels_mask(part)
        hflip = self.random_flip and np.random.binomial(1, p=0.5)
        if hflip:
            img, mask = map(Fvision.hflip, [img, mask])
            size = EVAL_IMG_SIZE[0] if self.eval_mode else self.img_size[0]

        #img = self.transform(img)
        #mask = self.transform_mask(mask)
        #part = self.transform(part_img)
        img = to_tensor(img)
        mask = to_tensor(mask)
        img = img * mask + torch.ones_like(img) * (1 - mask)
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': img, 'masks': mask, 'parts': part, 'poses': poses}, {'kps': -1}

    @property
    @lru_cache()
    def transform(self):
        return Compose([Resize(self.img_size[0]), ToTensor()])

    @property
    @lru_cache()
    def transform_mask(self):
        size = EVAL_IMG_SIZE[0] if self.eval_mode else self.img_size[0]
        return Compose([Resize(size, interpolation=InterpolationMode.NEAREST), ToTensor()])

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
    
    def assign_pixels_to_cluster(self,image):
        #image = Image.open(image_path)
        pixels = np.array(image).reshape(-1,3).astype(np.float32)
        centers = np.array(color_map,dtype=np.float32)
        distances = np.linalg.norm(pixels[:,np.newaxis]-centers,axis=2)
        labels = np.argmin(distances,axis=1)
        return labels.reshape(image.size[0],image.size[1])
    
    def get_models(self):
        split = self.split
        f = open(self.data_path.parent / f'{split}.lst')
        names = f.read().split("\n")
        return names
        
    @torch.no_grad()
    def convert_labels_to_colors(self,labels,filename):
        if labels.shape[-1] != N_K+1:
            labels = labels.permute(1,2,0)
        labels, _ = labels.split([N_K,1],dim=-1)
        labels = labels.unsqueeze(3)
        colors = torch.mul(labels.repeat(1,1,1,3).cpu(),torch.tensor(color_map[:N_K])).sum(dim=2)
        colors = colors.type(torch.uint8).numpy()
        im = Image.fromarray(colors).convert('RGB')
        im.save(filename)
        
    def convert_labels_mask(self,labels):
        if labels.shape[-1] != N_K+1:
            labels = labels.permute(1,2,0)
        labels, _ = torch.tensor(labels).split([N_K,1],dim=-1)
        labels = labels.unsqueeze(3)
        colors = torch.mul(labels.repeat(1,1,1,1).cpu(),torch.tensor(mask_map[:N_K])).sum(dim=2)
        colors = colors.type(torch.uint8).numpy()
        im = Image.fromarray(colors).convert('L')
        return im