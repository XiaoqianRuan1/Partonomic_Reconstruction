# Learning Partonomic 3D Reconstruction from Image Collections
This is an implementation of the CVPR25 paper ["Learning Partonomic 3D Reconstruction from Image Collections"](https://openaccess.thecvf.com/content/CVPR2025/papers/Ruan_Learning_Partonomic_3D_Reconstruction_from_Image_Collections_CVPR_2025_paper.pdf). 


## Installation
### Create conda environment
```bash
conda env create -f environment.yml
conda activate part
```
## Datasets
Please download our generated dataset at the following links:
| Dataset | Link |
|:----------:|:----------:|
| ShapeNetPart | [ShapeNetPart](https://huggingface.co/datasets/xiaoqian12/Partonomic/blob/main/ShapeNetPart.zip) |
| PartNet | [PartNet](https://huggingface.co/datasets/xiaoqian12/Partonomic/blob/main/PartNet.zip) |
| CUB-200-2011 | [CUB-200-2011](https://huggingface.co/datasets/xiaoqian12/Partonomic/blob/main/cub.zip) |
## Training
To launch a training from scratch, run
```bash
cuda=gpu_id config=filename.yml tag=run_tag ./script/pipeline.sh
```
where gpu_id is a device id, filename.yml is a config folder, run_tag is a tag for the experiment. 
Results are saved at ```runs/${DATASET}/${DATE}_${run_tag}``` where ```DATASET``` is the dataset name specified in ```filename.yml``` and ```DATE``` is the current date in ```mmdd``` format. 

### Configs and guidences
- ```snp/*.yml``` for ShapeNetPart category

- ```part/*.yml``` for PartNet category

- ```cub.yml``` for CUB-200-2011 dataset

## Evaluation
A model is evaluated at the end of training. To evaluate a pretrained model
- Create a new folder and move this model
- Modify the filename.yml by pointing the folder under ```${resume}```
- lanuch the evaluation
```bash
cuda=gpu_id config=snp/car.yml tag=car_eval ./script/pipeline.sh
```
## References
If you find our work helpful, please consider citing our paper:
```bash
@inproceedings{ruan2025learning,
  title={Learning Partonomic 3D Reconstruction from Image Collections},
  author={Ruan, Xiaoqian and Yu, Pei and Jia, Dian and Park, Hyeonjeong and Xiong, Peixi and Tang, Wei},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={26734--26744},
  year={2025}
}
```

## Acknowledge
Our project learns from [Unicorn](https://arxiv.org/pdf/2204.10310) and [AST](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03170.pdf).
