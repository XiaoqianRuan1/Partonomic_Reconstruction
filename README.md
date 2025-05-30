# 3D Partonomic Reconstruction
  
![Qualitative Results](teaser.gif)
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
cuda=gpu_id config=snp/airplane.yml tag=airplane_eval ./script/pipeline.sh
```
