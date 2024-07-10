# cv4lymphnode
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12702916.svg)](https://doi.org/10.5281/zenodo.12702916)

## python requirements
```
torch
efficientnet_pytorch
torchvision
timm
pytorch_grad_cam
cv2
PIL
sklearn
```

## download dataset
please download dataset at https://zenodo.org/records/12702916, and unzip it at dataset folder

## train
```
python main.py
```

## metric
```
python metric.py
```
then will generate metric.png


## extract
use pytorch_grad_cam to generate image importance heatmap
```
python extract.py
```

## predict
you can change image_path and model_path at predictor.py
```
python predictor.py
```
