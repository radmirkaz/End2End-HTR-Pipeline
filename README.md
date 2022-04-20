# End2End-HTR-Pipeline

This repository contains all files to train end-to-end recognition of handwritten words from school notebooks in Russian and English. With the end-to-end system, you can convert your school notebook to text without quality loss. This repository does not include any models. 

A good [dataset](https://github.com/abdoelsayed2016/HKR_Dataset) to train the model.

## Pipeline

The pipeline is made up of several blocks:
![blocks](https://github.com/RadmirZ/End2End-HTR-Pipeline/blob/main/diagram.png)
- Input: school notebook's page image
- Image segmentation: we use Detectron2 to detect and segment each word from the image
- OCR: after segmentation we predict each detected word separately
- Beam search (will be added): final decision making layer to choose the best output given target variables
- Output: predicted text

## Setup

Use config.py to quickly set parameters and train the model. We used [omegaconf](https://github.com/omry/omegaconf) for more convenience.

To train models you should change paths in config files (in case of ocr model you should checge necessary parameters)<br>
```jsonc
'paths': {
  'path_to_images': 'your/path/images/',
  'path_to_external_images': '',

  'path_to_csv': 'your/path/train.csv',

  'path_to_checkpoints': './submission/checkpoints/',
  'log_name': 'log.txt',
},
```

## Visualizations

![sample1](https://github.com/RadmirZ/End2End-HTR-Pipeline/blob/main/example1.jpg)
![sample2](https://github.com/RadmirZ/End2End-HTR-Pipeline/blob/main/example2.jpg)

