import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from utils import path_exists
import os
import cv2
import matplotlib.pyplot as plt

DATASETS_PATH = "/data/unicorn-main1/datasets/shapenet_nmr/"
part_root = "/data/unicorn-part/data/ShapeNetPart/"
#part_root = "/mnt/sde1/xiaoqianruan/unicorn-main_2d_ground_truth/data/ShapeNetPart/ground/"

color_map = [
    [255,0,0],
    [0,0,255],
    [0,255,0],
    [200,0,255],
    [255,255,255],
]

color_map_new = [
    [255,0,0],
    [0,0,255],
    [0,255,0],
    [200,0,255],
    [0,0,0],
]

def translate_images(mask_image_path,part_image_path):
    translation_folder = "/home/xruan9/data/unicorn-part/data/ShapeNetPart/aligned/"
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    part_image = cv2.imread(part_image_path, cv2.IMREAD_UNCHANGED)
    model = part_image.split("/")[-2]
    image_name = part_image.split("/")[-1]
    
    # Convert mask image to grayscale (if it's not already)
    mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY) if len(mask_image.shape) == 3 else mask_image
    
    # Calculate moments to find the centroid of the objects in both images
    moments_mask = cv2.moments(mask_image_gray)
    moments_part = cv2.moments(cv2.cvtColor(part_image, cv2.COLOR_BGR2GRAY))
    
    # Calculate centroids
    centroid_mask = (int(moments_mask["m10"] / moments_mask["m00"]), int(moments_mask["m01"] / moments_mask["m00"]))
    centroid_part = (int(moments_part["m10"] / moments_part["m00"]), int(moments_part["m01"] / moments_part["m00"]))
    
    # Calculate translation vector
    translation_vector = (centroid_mask[0] - centroid_part[0], centroid_mask[1] - centroid_part[1])
    
    # Translate the part image
    translation_matrix = np.float32([[1, 0, translation_vector[0]], [0, 1, translation_vector[1]]])
    translated_part_image = cv2.warpAffine(part_image, translation_matrix, (part_image.shape[1], part_image.shape[0]))
    
    translated_image_path = os.path.join(translation_folder,model,image_name)
    cv2.imwrite(translated_image_path, translated_new_part_image)
    
def read_mask_part_image():
    mask_folder = "/home/xruan9/data/unicorn-main1/datasets/shapenet_nmr/shapenet_nmr/"
    part_folder = "/home/xruan9/data/unicorn-part/data/ShapeNetPart/"
    #category = ["02691156", "02958343", "03001627", "03636649", "04379243"]
    category = "04379243"
    mask_folder = os.path.join(mask_folder, category)
    part_folder = os.path.join(part_folder, category)
    mask_images = glob.glob(mask_foder+"/*/mask/*.png")
    for img in mask_images:
        img_name = img.split("/")[-1]
        print(img_name)
        model = img.split("/")[-3]
        print(model)
        part_image = os.path.join(part_folder+str(model), img_name)
        print(part_image)
        print(aa)
        translate_images(img,part_image)

if __name__ == "__main__":
    read_mask_part_image()