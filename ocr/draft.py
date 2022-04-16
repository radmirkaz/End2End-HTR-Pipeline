import enum
import os
import json
import random
from scipy.sparse.construct import rand
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import albumentations as A

import cv2
import timm
import torch
import pickle

import matplotlib
import matplotlib.pyplot as plt
#
# from config import config
# from custom.augmentations import ExtraLinesAugmentation

import difflib
import time
#
from config import config
# from data import get_loaders, get_loader_inference
# from custom.scheduler import GradualWarmupSchedulerV2
# from custom.augmentations import cutmix, mixup
# from custom.models import Seq2SeqModel
# from custom.metrics import cer
import re
from collections import Counter
from skimage.transform import resize
from utils import *
#
# def words(text):
#     return re.findall(r'\w+', text.lower())
#
# with open('words.pkl', 'rb') as f:
#     words_data = pickle.load(f)
#
# WORDS = Counter(words_data)
#
# def P(word, N=sum(WORDS.values())):
#     "Probability of `word`."
#     return WORDS[word] / N
#
# def correction(word):
#     "Most probable spelling correction for word."
#     return max(candidates(word), key=P)
#
#
# def candidates(word):
#     "Generate possible spelling corrections for word."
#     return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
#
#
# def known(words):
#     "The subset of `words` that appear in the dictionary of WORDS."
#     return set(w for w in words if w in WORDS)
#
#
# def edits1(word):
#     "All edits that are one edit away from `word`."
#     letters = 'abcdefghijklmnopqrstuvwxyz'
#     splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#     deletes = [L + R[1:] for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
#     replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
#     inserts = [L + c + R for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)
#
#
# def edits2(word):
#     "All edits that are two edits away from `word`."
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))
#
#
# def get_inference_model(config):
#     '''Get PyTorch model.'''
#
#     model_name = config.model.name
#     checkpoint_path = config.model.checkpoint_path
#
#     if model_name.startswith('/custom/'):
#         model = globals()[model_name[8:]](config)
#     else:
#         raise RuntimeError('Unknown model source. Use /custom/.')
#
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
#     if 'model' in checkpoint:
#         model.load_state_dict(checkpoint['model'])
#         print(f'Loaded {model_name}')
#     else:
#         model.load_state_dict(checkpoint)
#
#     model = model.to('cuda')
#
#     return model
#
#
# def validation(config, model, val_loader, loss_function, words):
#     '''Validation loop.'''
#
#     print('Validating')
#
#     with open('data/idx2text_dict.pkl', 'rb') as f:
#         idx2text_dict = pickle.load(f)
#     with open('data/text2idx_dict.pkl', 'rb') as f:
#         text2idx_dict = pickle.load(f)
#
#     model.eval()
#
#     total_loss = 0.0
#     total_metric = 0.0
#
#     preds, targets = [], []
#
#     with torch.no_grad():
#         for step, batch in enumerate(tqdm(val_loader)):
#             inputs, targets = batch
#             inputs, targets = inputs.to(config.training.device), targets.to(config.training.device)
#
#             # Letter-by-letter evaluation
#             cnn_output = model.backbone(inputs.float())
#             cnn_output = cnn_output.flatten(2).permute(2, 0, 1)
#             memory = model.transformer.encoder(model.pos_encoder(cnn_output).permute(1, 0, 2))
#
#             prob_values = 1
#             out_indexes = [text2idx_dict['SOS'], ]
#             for x in range(100):
#                 target_tensor = torch.LongTensor(out_indexes).unsqueeze(0).to(config.training.device)
#
#                 output = model.decoder(target_tensor)
#                 output = model.pos_decoder(output)
#                 output = model.transformer.decoder(output, memory)
#                 output = model.out(output)
#
#                 output_token = torch.argmax(output, dim=2)[:, -1].item()
#
#                 prob_values = prob_values * torch.sigmoid(output[-1, 0, output_token]).item()
#
#                 out_indexes.append(output_token)
#
#                 if output_token == text2idx_dict['EOS']:
#                     break
#
#             # Decode indexes
#             vec_func = lambda x: idx2text_dict[x]
#             idx2text_func = np.vectorize(vec_func)
#             outputs = np.expand_dims(idx2text_func(out_indexes), 0)
#             targets = idx2text_func(targets.to('cpu').numpy())
#
#             # Ensure that we have EOS token in the back
#             outputs = np.append(outputs, [['EOS']], axis=1)
#
#             # outputs = model(inputs.float(), targets)
#
#             # if step == 0:
#             #     plt.subplot(1, 2, 1)
#             #     plt.imshow(masks.to('cpu').numpy()[0, :, :, :].transpose(1, 2, 0))
#
#             #     plt.subplot(1, 2, 2)
#             #     plt.imshow(outputs.to('cpu').numpy()[0, :, :, :].transpose(1, 2, 0))
#
#             #     plt.show()
#
#             # loss = loss_function(outputs, targets)
#             # total_loss += loss.item()
#
#             # outputs = torch.argmax(outputs, dim=1).to('cpu').numpy()
#
#             # # Decode indexes
#             # vec_func = lambda x: idx2text_dict[x]
#             # idx2text_func = np.vectorize(vec_func)
#             # outputs = idx2text_func(outputs)
#             # targets = idx2text_func(targets.to('cpu').numpy())
#
#             # Acquire strings
#             targets = np.apply_along_axis(lambda x: ''.join(x.tolist()[1:x.tolist().index('EOS')]), 1, targets)
#             outputs = np.apply_along_axis(lambda x: ''.join(x.tolist()[1:x.tolist().index('EOS')]), 1, outputs)
#
#             if step == 1:
#                 print(targets)
#                 print(outputs)
#                 print(correction(outputs[0]))
#
#             # outputs = correction(outputs[0])
#
#             metric = get_metric(config, targets, outputs)
#             total_metric += metric
#
#     return total_loss / len(val_loader), total_metric / len(val_loader)
#
#
# def get_metric(config, y_true, y_pred):
#     '''Calculate metric.'''
#
#     predictions = y_pred
#
#     if config.metric.name.startswith('/custom/'):
#         score = globals()[config.metric.name[8:]](y_true, predictions, **config.metric.params)
#     else:
#         score = getattr(sklearn.metrics, config.metric.name)(y_true, predictions, **config.metric.params)
#
#     return score
from collections import Counter
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import cv2
#
# # convert images and labels into defined data structures
# def process_data(image_dir, labels_dir, ignore=[]):
#     """
#     params
#     ---
#     image_dir : str
#       path to directory with images
#     labels_dir : str
#       path to tsv file with labels
#     returns
#     ---
#     img2label : dict
#       keys are names of images and values are correspondent labels
#     chars : list
#       all unique chars used in data
#     all_labels : list
#     """
#
#     chars = []
#     img2label = dict()
#
#     raw = open(labels_dir, 'r', encoding='utf-8').read()
#     temp = raw.split('\n')
#     for t in temp:
#         try:
#             x = t.split('\t')
#             flag = False
#             for item in ignore:
#                 if item in x[1]:
#                     flag = True
#             if flag == False:
#                 img2label[image_dir + x[0]] = x[1]
#                 for char in x[1]:
#                     if char not in chars:
#                         chars.append(char)
#         except:
#             print('ValueError:', x)
#             pass
#
#     all_labels = list(img2label.values())
#     chars.sort()
#
#     return img2label, chars, all_labels
#
#
# # GENERATE IMAGES FROM A FOLDER
# def generate_images(img_paths):
#     """
#     params
#     ---
#     names : list of str
#         paths to images
#     returns
#     ---
#     data_images : list of np.array
#         images in np.array format
#     """
#     data_images = []
#     for path in tqdm(img_paths):
#         img = cv2.imread(path)
#         try:
#             data_images.append(img.astype('uint8'))
#         except:
#             img = process_image(img)
#     return data_images


# from config import config
import sys
from data import get_loaders

if __name__ == "__main__":
    model = get_model(config)
    train_loader, val_loader = get_loaders(config)

    val_loss, current_metric = validation(config, model, val_loader, None)

    print(current_metric)






