import argparse
import warnings

import numpy as np
from torch.utils.data import DataLoader

from dataset import get_dataset
from model import load_model_from_path
from model.renderer import save_mesh_as_gif
from utils import path_mkdir
from utils.path import MODELS_PATH
from utils.logger import print_log
from utils.mesh import save_mesh_as_obj, normalize
from utils.pytorch import get_torch_device
import os
from utils.metrics import MeshEvaluator, ProxyEvaluator,PartEvaluator
from model.unicorn_deform import PartMesh
import torch
from utils.pytorch import torch_to
from utils.image import convert_to_img

BATCH_SIZE = 4
N_WORKERS = 4
PRINT_ITER = 2
SAVE_GIF = True
warnings.filterwarnings("ignore")

N_K = 4
color_map = [
    [255,0,0],
    [0,0,255],
    [0,255,0],
    #[200,0,255],
    [128,0,128],
    [0,0,0],
]


@torch.no_grad()
def generate_reconstruction(args):
    device = get_torch_device()
    m = load_model_from_path(MODELS_PATH / args.model).to(device)
    m.eval()
    print_log(f"Model {args.model} loaded: input img_size is set to {m.init_kwargs['img_size']}")
    
    # read the first image from the folder
    data = get_dataset(args.input)(split='test', img_size=m.init_kwargs['img_size'], categories=args.category)
    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)
    print_log(f"Found {len(data)} images in the folder")
    
    print_log("Starting reconstruction ....")
    out = path_mkdir(args.output)
    evaluator = PartEvaluator()
    txt_path = os.path.join(out,"results.txt")
    name_path = os.path.join(out,"name.txt")
    for j, (inp, labels) in enumerate(loader):
        imgs = inp['imgs'].to(device)
        meshes, part_meshes, (R, T), bkgs = m.predict_mesh_pose_bkg(imgs)
        #meshes, part_meshes, (R, T), slcys, bkgs = m.predict_mesh_pose_slcy_bkg(imgs)
        if not torch.all(inp["poses"] == -1):
            verts, faces = part_meshes.verts_padded(), part_meshes.faces_padded()
            R_gt, T_gt = map(lambda t: t.squeeze(2), inp["poses"].to(device).split([3,1],dim=2))
            verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
        part_meshes = PartMesh(verts=verts, faces=faces, labels=part_meshes.vert_labels)
        evaluator.results_output(part_meshes, torch_to(labels, device),txt_path)
        B, d, e = len(imgs), m.T_init[-1], np.mean(m.elev_range)
        for k in range(B):
            nb = j*B + k
            if nb % PRINT_ITER == 0:
                print_log(f"Reconstructed {nb} images...")
            name = data.models[nb]['model']
            f = open(name_path,"a")
            f.writelines(str(name))
            f.writelines("\n")
            convert_to_img(imgs[k]).save(out / f'{name}_inpraw.png')
            mcenter = normalize(meshes[k])
            mcenter_part = normalize(part_meshes[k])
            mcenter_part.textures = m.get_outputs_colors(color_map,k)
            save_mesh_as_obj(mcenter, out / f'{name}_mesh.obj')
            save_mesh_as_obj(mcenter_part, out / f'{name}_mesh_part.obj')
            if SAVE_GIF:
                save_mesh_as_gif(mcenter_part, out / f'{name}_mesh_part.gif', n_views=100, dist=d, elev=e, renderer=m.layer_renderer)

    print_log("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D reconstruction from single-view images in a folder')
    parser.add_argument('-c', '--category', nargs='?', type=str, required=True, help='dataset category')
    parser.add_argument('-m', '--model', nargs='?', type=str, required=True, help='Model name to use')
    parser.add_argument('-i', '--input', nargs='?', type=str, required=True, help='TestDataset in folder') 
    parser.add_argument('-o', '--output', nargs='?', type=str, required=True, help='output')
    args = parser.parse_args()
    assert args.model is not None and args.input is not None
    generate_reconstruction(args)
    """
    # read the model from the folder
    device = get_torch_device()
    m = load_model_from_path(MODELS_PATH / args.model).to(device)
    m.eval()
    print_log(f"Model {args.model} loaded: input img_size is set to {m.init_kwargs['img_size']}")
    
    # read the first image from the folder
    data = get_dataset(args.input)(split='test', img_size=m.init_kwargs['img_size'], categories=args.category)
    loader = DataLoader(data, batch_size=BATCH_SIZE, num_workers=N_WORKERS, shuffle=False)
    print_log(f"Found {len(data)} images in the folder")
    
    print_log("Starting reconstruction ....")
    out = path_mkdir(args.output)
    evaluator = PartEvaluator()
    txt_path = os.path.join(out,"results.txt")
    
    for j, (inp, labels) in enumerate(loader):
        imgs = inp['imgs'].to(device)
        meshes, part_meshes, (R, T), bkgs = m.predict_mesh_pose_bkg(imgs)
        if not torch.all(inp["poses"] == -1):
            verts, faces = part_meshes.verts_padded(), part_meshes.faces_padded()
            R_gt, T_gt = map(lambda t: t.squeeze(2), inp["poses"].to(device).split([3,1],dim=2))
            verts = (verts @ R + T[:, None] - T_gt[:, None]) @ R_gt.transpose(1, 2)
        part_meshes = PartMesh(verts=verts, faces=faces, labels=part_meshes.vert_labels)
        evaluator.results_output(part_meshes, torch_to(labels, device),txt_path)
        B, d, e = len(imgs), m.T_init[-1], np.mean(m.elev_range)
        for k in range(B):
            nb = j*B + k
            if nb % PRINT_ITER == 0:
                print_log(f"Reconstructed {nb} images...")
            name = data.input_files[nb].stem
            print(name)
            print(aa)
            convert_to_img(imgs[k]).save(out / f'{name}_inpraw.png')
            mcenter = normalize(meshes[k])
            mcenter_part = normalize(part_meshes[k])
            mcenter_part.textures = m.get_outputs_colors(color_map,k)
            save_mesh_as_obj(mcenter, out / f'{name}_mesh.obj')
            save_mesh_as_obj(mcenter_part, out / f'{name}_mesh_part.obj')
            if SAVE_GIF:
                save_mesh_as_gif(mcenter_part, out / f'{name}_mesh.gif', n_views=100, dist=d, elev=e, renderer=m.layer_renderer)

    print_log("Done!")
    """