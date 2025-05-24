import glob
from PIL import Image
import os
import numpy as np

color_list = {"beak":[0,0,255],
              "head": [0,0,255],
              "eye": [0,0,255],
              "body": [0,255,0],
              "wing": [255,0,0],
              "tail": [200,0,255],
              "neck": [0,255,255],
              "leg": [255,255,0],
              "background": [0,0,0]}


def read_images(file_path):
    part_name = file_path.split(".")[0].split("_")[-1]
    part_size = file_path.split("_")[-2]
    if part_size == "left" or part_size == "right":
        file_name_list = file_path.split("_")[:-2]
        file_name = "_".join(map(str,file_name_list))
    else:
        file_name_list = file_path.split("_")[:-1]
        file_name = "_".join(map(str,file_name_list))
    
def check_size(image_path,part_path):
    image = Image.open(image_path).convert('RGB')
    part = Image.open(part_path).convert('RGB')
    if image.size == part.size:
        print(True)
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!")




   
if __name__ == "__main__":
    for i in range(1,71):
        folder_name = str(i)
        file_folder = "/data/unicorn-part/data/AnnotationMasksPerclass/"
        if not os.path.exists(os.path.join(file_folder,folder_name)):
            continue
        image_list = glob.glob(os.path.join(file_folder,folder_name,"*.png"))
        image_list = sorted(image_list)
        new_image_number = []
        mask_files = []
        #output_folder = os.path.join("/data/unicorn-part/data/cub",folder_name)
        name_length = len(image_list[0].split("_"))
        folder_list = image_list[0].split("/")[-1].split("_")[:name_length-3]
        folder = "_".join(map(str,folder_list))
        folder_name = str(i).zfill(3)+"."+folder
        output_folder = os.path.join("/data/unicorn-part/data/cub_new",folder_name)
        image_path = "/data/unicorn-main1/datasets/cub_200/images/"
        image_folder = os.path.join(image_path,folder_name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for image_name in image_list:
            image_number = image_name.split("_")[name_length-3]
            image_path = image_name.split("/")[-1]
            list_number_list = image_path.split("_")[:name_length-1]
            list_number = "_".join(map(str,list_number_list))
            #list_number = image_name.split("_")[name_length-2]
            if len(new_image_number) == 0:
                new_image_number.append(image_number)
                mask_files.append(image_name)
            elif image_number not in new_image_number:
                new_image_number.append(image_number)
                sample_mask = np.array(Image.open(mask_files[0]).convert('L'))
                combined_image = np.zeros((sample_mask.shape[0],sample_mask.shape[1],3),dtype=np.uint8)
                mask_number_list = mask_files[0].split("/")[-1].split("_")[:name_length-1]
                mask_number = "_".join(map(str,mask_number_list))
                print(mask_number)
                for mask_file in mask_files:
                    mask = np.array(Image.open(mask_file).convert('L'))
                    part_number = mask_file.split("_")[-1].split(".")[0]
                    color = color_list[part_number]
                    combined_image[mask>0] = color
                output_image = Image.fromarray(combined_image)
                output_path = os.path.join(output_folder,str(mask_number)+".jpg")
                #image_paths = os.path.join(image_folder,str(mask_number)+".jpg")
                output_image.save(output_path)
                #check_size(image_paths,output_path)
                mask_files = []
                mask_files.append(image_name)
            else:
                mask_files.append(image_name)           
            