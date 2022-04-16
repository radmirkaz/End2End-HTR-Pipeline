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
