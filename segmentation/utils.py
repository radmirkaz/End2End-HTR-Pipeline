import os
import gc
import cv2
import numpy as np
from tqdm import tqdm

import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, DefaultPredictor

import matplotlib.pyplot as plt

from config import config
from data import *
from metrics import *


def create_detectron_config(config):
    detectron_config = get_cfg()
    detectron_config.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    detectron_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml')
    # detectron_config.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    # detectron_config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    
    detectron_config.SEED = config.general.seed

    detectron_config.DATASETS.TRAIN = ('dataset_train', )
    detectron_config.DATASETS.TEST = ('dataset_val', )

    detectron_config.INPUT.MIN_SIZE_TRAIN = 1300
    detectron_config.INPUT.MAX_SIZE_TRAIN = 1300

    # detectron_config.INPUT.RANDOM_FLIP = 'vertical'
    
    detectron_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    detectron_config.MODEL.ROI_HEADS.NUM_CLASSES = 1
    detectron_config.MODEL.ROI_BOX_HEAD.NORM = "BN"
    detectron_config.MODEL.ROI_MASK_HEAD.NORM = "BN"
    
    detectron_config.INPUT.FORMAT = 'BGR'
    detectron_config.DATALOADER.NUM_WORKERS = 4

    # detectron_config.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'
    # detectron_config.SOLVER.BASE_LR_END = 0.0
    detectron_config.SOLVER.IMS_PER_BATCH = 2
    detectron_config.SOLVER.BASE_LR = 0.01 
    detectron_config.SOLVER.GAMMA = 1.0 
    # detectron_config.SOLVER.STEPS = (1500,)
    detectron_config.SOLVER.MAX_ITER = 15000
    detectron_config.SOLVER.CHECKPOINT_PERIOD = 1000

    detectron_config.OUTPUT_DIR = config.paths.path_to_checkpoints

    os.makedirs(detectron_config.OUTPUT_DIR, exist_ok=True)

    print('Detectron config created')

    return detectron_config


def train(config, detectron_config):
    trainer = DefaultTrainer(detectron_config)
    
    trainer.resume_or_load(resume=False)
    trainer.train()


def validation(config, detectron_config):
    # Validation Detectron config
    detectron_config.MODEL.WEIGHTS = config.paths.path_to_checkpoints + '/model_final.pth'
    #detectron_config.MODEL.WEIGHTS = config.paths.path_to_checkpoints + '/model_0013999.pth'
    detectron_config.DATASETS.TEST = ('dataset_val', )

    detectron_config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    detectron_config.INPUT.MIN_SIZE_TEST = 1300
    detectron_config.INPUT.MAX_SIZE_TEST = 1300
    detectron_config.INPUT.FORMAT = 'BGR'

    detectron_config.TEST.DETECTIONS_PER_IMAGE = 1000

    # Predictor
    predictor = DefaultPredictor(detectron_config)

    # Load validation images and annotations
    with open(config.paths.path_to_active_training_data + '/val/annotations_new.json') as f:
        annotations_val = json.load(f)

    val_images = annotations_val['images']

    # Make predictions
    val_predictions = {}
    for val_img in tqdm(val_images):
        file_name = val_img['file_name']

        img_path = os.path.join(config.paths.path_to_active_training_data + '/val/images/', file_name)
        img = cv2.imread(img_path)

        outputs = predictor(img)
        prediction = outputs['instances'].pred_masks.cpu().numpy()

        mask = np.add.reduce(prediction)
        mask = mask > 0

        val_predictions[file_name] = mask

    # Compute metrics
    train_targets = np.load('data/binary.npz')

    metrics = []
    for key in tqdm(val_predictions.keys()):
        pred = val_predictions[key].reshape(-1)
        true = train_targets[key].reshape(-1)

        f1 = compute_f1(true, pred)
        metrics.append(f1)

    print(f'Mean metric: {np.mean(metrics)}')


def run(config):
    create_train_and_val_images_and_annotations(config)
    create_dataset_catalogs(config)

    detectron_config = create_detectron_config(config)

    train(config, detectron_config)

    validation(config, detectron_config)
