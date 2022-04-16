import numpy as np
import pandas as pd

import os
import gc
import cv2
import json
import pickle
import shutil
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from custom.augmentations import ExtraLinesAugmentation

from custom.augmentations import *

import matplotlib.pyplot as plt


class OCRDataset(Dataset):
    def __init__(self, config, df, text2idx_dict, is_eval, transfroms=None):
        self.config = config

        self.df = df
        self.is_eval = is_eval
        self.transforms = transfroms
        self.text2idx_dict = text2idx_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample_path = self.config.paths.path_to_images + self.df[self.config.data.id_column].iloc[idx]

        if '.png' in self.df[self.config.data.id_column].iloc[idx]:
            if self.df['source'].iloc[idx] == 'original':
                img = cv2.imread(self.config.paths.path_to_images + self.df[self.config.data.id_column].iloc[
                    idx] + self.config.data.image_format)
            elif self.df['source'].iloc[idx] == 'external':
                if self.df['num'].iloc[idx] not in [3, 4]:
                    img = cv2.imread(
                        self.config.paths.path_to_external_images2 + self.df[self.config.data.id_column].iloc[
                            idx] + self.config.data.image_format)
                elif self.df['num'].iloc[idx] == 4:
                    img = cv2.imread(
                        self.config.paths.path_to_external_images4 + self.df[self.config.data.id_column].iloc[
                            idx] + self.config.data.image_format)
                else:
                    img = cv2.imread(
                        self.config.paths.path_to_external_images3 + self.df[self.config.data.id_column].iloc[
                            idx] + self.config.data.image_format)
        elif '.jpg' in self.df[self.config.data.id_column].iloc[idx]:
            img = cv2.imread(self.config.paths.path_to_external_images + self.df[self.config.data.id_column].iloc[
                idx] + self.config.data.image_format)

        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            img = np.zeros((128, 384))

        if not self.config.inference.inference:
            label = self.df[self.config.data.target_columns].iloc[idx].values[0]
            label = self.text_to_indexes(label, self.text2idx_dict)

            if self.transforms:
                img = self.transforms(image=img)['image']

            return img, torch.LongTensor(label)
        else:
            if self.transforms:
                img = self.transforms(image=img)['image']

            return torch.FloatTensor(img)

    def text_to_indexes(self, label, text2idx_dict):
        return [text2idx_dict['SOS']] + [text2idx_dict[i] for i in label if i in text2idx_dict.keys()] + [
            text2idx_dict['EOS']]


class TextCollate:
    def __call__(self, batch):
        x_padded = []
        max_y_len = max([i[1].size(0) for i in batch])

        y_padded = torch.LongTensor(max_y_len, len(batch))
        y_padded.zero_()

        for i in range(len(batch)):
            x_padded.append(batch[i][0].unsqueeze(0))

            y = batch[i][1]
            y_padded[:y.size(0), i] = y

        x_padded = torch.cat(x_padded)
        y_padded = y_padded.transpose(0, 1)

        return x_padded, y_padded


def get_transforms(config):
    '''Get train and validation augmentations.'''

    pre_transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                      else getattr(A, item['name'])(**item['params']) for item in config.augmentations.pre_transforms]
    transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                  else getattr(A, item['name'])(**item['params']) for item in config.augmentations.transforms]
    post_transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                       else getattr(A, item['name'])(**item['params']) for item in config.augmentations.post_transforms]

    train_transforms = A.Compose(pre_transforms + transforms + post_transforms)
    valid_transforms = A.Compose(pre_transforms + post_transforms)

    return train_transforms, valid_transforms


def data_generator(config):
    '''Generate data for train and validation splits.'''

    print('Getting the data')

    assert abs(
        config.data.train_size + config.data.val_size + config.data.test_size - 1.0) < 1e-9, 'sum of the sizes of splits must be equal to 1.0'

    data = pd.read_csv(config.paths.path_to_csv)
    data = data.loc[data['source'] == 'original']

    if config.training.debug:
        data = data.sample(n=config.training.debug_number_of_samples, random_state=config.general.seed).reset_index(
            drop=True)

    if config.data.kfold.use_kfold:
        kfold = getattr(model_selection, config.data.kfold.name)(**config.data.kfold.params)

        if config.data.kfold.group_column:
            groups = data[config.data.kfold.group_column]
        else:
            groups = None

        for fold, (train_index, val_index) in enumerate(
                kfold.split(data, data[config.data.kfold.split_on_column], groups)):
            if fold == config.data.kfold.current_fold:
                train_images = data[config.data.id_column].iloc[train_index].values
                train_targets = None
                val_images = data[config.data.id_column].iloc[val_index].values
                val_targets = None

                break

        if config.data.test_size == 0.0:
            return train_images, train_targets, val_images, val_targets

        val_size = config.data.val_size / (config.data.val_size + config.data.test_size)
        test_size = config.data.test_size / (config.data.val_size + config.data.test_size)
        val_images, test_images, val_targets, test_targets = train_test_split(val_images, val_targets,
                                                                              train_size=val_size,
                                                                              test_size=test_size,
                                                                              random_state=config.general.seed,
                                                                              stratify=val_targets)

        return train_images, train_targets, val_images, val_targets, test_images, test_targets
    else:
        pass


def get_loaders(config):
    '''Get data loaders.'''

    train_transforms, val_transforms = get_transforms(config)
    df = pd.read_csv(config.paths.path_to_csv)

    # Letters
    counter = Counter(''.join(df['label'].values))
    letters = ['PAD', 'SOS'] + sorted(list(set(counter.keys()))) + ['EOS']

    # Indexes for letters
    text2idx_dict = {p: idx for idx, p in enumerate(letters)}
    idx2text_dict = {idx: p for idx, p in enumerate(letters)}
    print(text2idx_dict)

    with open('data/text2idx_dict.pkl', 'wb') as f:
        pickle.dump(text2idx_dict, f)
    with open('data/idx2text_dict.pkl', 'wb') as f:
        pickle.dump(idx2text_dict, f)

    shutil.copyfile('data/text2idx_dict.pkl', 'submission/text2idx_dict.pkl')
    shutil.copyfile('data/idx2text_dict.pkl', 'submission/idx2text_dict.pkl')

    # Get loaders
    if config.data.test_size == 0.0:
        train_images, _, val_images, _ = data_generator(config)

        train_images = np.concatenate([train_images, df[config.data.id_column].loc[df['source'] == 'external'].values])

        train_dataset = OCRDataset(config, df[df[config.data.id_column].isin(train_images)], text2idx_dict, False,
                                   train_transforms)
        val_dataset = OCRDataset(config, df[df[config.data.id_column].isin(val_images)], text2idx_dict, True,
                                 val_transforms)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.data.train_batch_size,
                                  pin_memory=False,
                                  num_workers=config.data.num_workers, collate_fn=TextCollate())
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                num_workers=config.data.num_workers, collate_fn=TextCollate())

        return train_loader, val_loader
    else:
        train_images, _, val_images, _, test_images, _ = data_generator(config)

        train_dataset = OCRDataset(config, df[df[config.data.id_column].isin(train_images)], text2idx_dict, False,
                                   train_transforms)
        val_dataset = OCRDataset(config, df[df[config.data.id_column].isin(val_images)], text2idx_dict, True,
                                 val_transforms)
        test_dataset = OCRDataset(config, df[df[config.data.id_column].isin(test_images)], text2idx_dict, True,
                                  val_transforms)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.data.train_batch_size,
                                  pin_memory=False,
                                  num_workers=config.data.num_workers, collate_fn=TextCollate())
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                num_workers=config.data.num_workers, collate_fn=TextCollate())
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.data.test_batch_size, pin_memory=False,
                                 num_workers=config.data.num_workers, collate_fn=TextCollate())

        return train_loader, val_loader, test_loader


def get_loader_inference(config):
    '''Get loader for the inference'''

    _, transforms = get_transforms(config)
    df = pd.read_csv(config.paths.path_to_sample_submission)

    dataset = OCRDataset(config, df, transforms)

    data_loader = DataLoader(dataset, shuffle=False, batch_size=config.inference.batch_size, pin_memory=True,
                             num_workers=config.data.num_workers, drop_last=False, collate_fn=TextCollate())

    return data_loader
