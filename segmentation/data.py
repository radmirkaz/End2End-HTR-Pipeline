import os
import json
import shutil

import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets import register_coco_instances, load_coco_json


def create_train_and_val_images_and_annotations(config):
    with open(config.paths.path_to_annotations_json) as f:
        annotations = json.load(f)

    # Create empty dictionaries
    annotations_val = {}
    annotations_val['categories'] = annotations['categories']

    annotations_train = {}
    annotations_train['categories'] = annotations['categories']

    # Split val and train
    annotations_val['images'] = []
    annotations_train['images'] = []
    for num,img in enumerate(annotations['images']):
        if num % 10 == 0:
            annotations_val['images'].append(img)
        else:
            annotations_train['images'].append(img)

    # Add ids
    val_img_id = [i['id'] for i in annotations_val['images']]
    train_img_id = [i['id'] for i in annotations_train['images']]
    
    # Add annotations
    annotations_val['annotations'] = []
    annotations_train['annotations'] = []

    for annot in annotations['annotations']:
        if annot['image_id'] in val_img_id:
            annotations_val['annotations'].append(annot)
        elif annot['image_id'] in train_img_id:
            annotations_train['annotations'].append(annot)
        else:
            print('Annotations not present in any of sets')

    # Create folders for val and train
    if not os.path.exists(config.paths.path_to_active_training_data + '/val/'):
        os.makedirs(config.paths.path_to_active_training_data + '/val/')
    if not os.path.exists(config.paths.path_to_active_training_data + '/val/images/'):
        os.makedirs(config.paths.path_to_active_training_data + '/val/images/')
    if not os.path.exists(config.paths.path_to_active_training_data + '/train/'):
        os.makedirs(config.paths.path_to_active_training_data + '/train/')
    if not os.path.exists(config.paths.path_to_active_training_data + '/train/images/'):
        os.makedirs(config.paths.path_to_active_training_data + '/train/images/')  

    # Save images and annotations for val and train
    for i in annotations_val['images']:
        shutil.copy(config.paths.path_to_images + '/' + i['file_name'], config.paths.path_to_active_training_data + '/val/images/')
    
    for i in annotations_train['images']:
        shutil.copy(config.paths.path_to_images + '/' +i['file_name'], config.paths.path_to_active_training_data + '/train/images/')

    with open(config.paths.path_to_active_training_data + '/val/annotations_new.json', 'w') as outfile:
        json.dump(annotations_val, outfile)
    
    with open(config.paths.path_to_active_training_data + '/train/annotations_new.json', 'w') as outfile:
        json.dump(annotations_train, outfile)

    print('### Train and val images and annotationgs are created ###')


def create_dataset_catalogs(config):
    for name in ['train', 'val']:
        DatasetCatalog.register('dataset_' + name, 
                                lambda name=name: load_coco_json(config.paths.path_to_active_training_data + '/{}/annotations_new.json'.format(name),
                                image_root=config.paths.path_to_images,
                                dataset_name='dataset_' + name,
                                extra_annotation_keys=['bbox_mode']))

    dataset_dictionaries_train = DatasetCatalog.get('dataset_train')
    train_metadata = MetadataCatalog.get('dataset_train')

    dataset_dictionaries_val = DatasetCatalog.get('dataset_val')
    val_metadata = MetadataCatalog.get('dataset_val')

    print(f'Size of train set: {len(dataset_dictionaries_train)}, size of val set: {len(dataset_dictionaries_val)}')




