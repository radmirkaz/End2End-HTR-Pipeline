import os
import pickle
import sys
import json
from typing import Tuple
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


from submission_config import config
from skimage.transform import resize

import torch
from tqdm import tqdm
import cv2
from PIL import Image

import torch.nn as nn
import torchvision
import timm

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from custom.models import *
from utils import get_inference_ocr_model, get_contours_from_mask, get_larger_contour, crop_img_by_polygon

from PIL import ImageFont, ImageDraw


def get_image_visualization(test_pr, img_name, output_path, path_to_font='arial.ttf', show=False):
    fontpath = path_to_font
    font_koef = 50
    pred_data = test_pr[img_name]

    img = cv2.imread(f'./segset2/train_segmentation/val/images/{img_name}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h / font_koef))
    img_overlay = Image.fromarray(img)
    draw = ImageDraw.Draw(img_overlay)

    for prediction in pred_data["predictions"]:
        polygon = prediction["polygon"]
        pred_text = prediction["text"]
        cv2.drawContours(img, np.array([polygon]), -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array([polygon]))
        draw.text((x, y - 30), pred_text, fill=0, font=font)

    #     vis_img = np.array(empty_img)
    #     vis = np.concatenate((img, vis_img), axis=1)

    plt.figure(figsize=(32, 14))
    plt.imshow(img_overlay)
    plt.savefig(f'{output_path}predicted_{img_name}')
    plt.show()


def main(config, test_images_path, output_path):
    # test_images_path, output_path = sys.argv[1:]
    #     if test_images_path is None:
    # test_images_path = './segset2/train_segmentation/val/images/'
    # output_path = 'predictions.json'
    os.makedirs(output_path, exist_ok=True)

    model = get_inference_ocr_model(config)
    model.eval()

    segm_predictor = SEGMpredictor(model_path=config.model.segm_model_path, config=config)

    with open(config.idx2text_dict_path, 'rb') as f:
        idx2text_dict = pickle.load(f)
    with open(config.text2idx_dict_path, 'rb') as f:
        text2idx_dict = pickle.load(f)

    f_predictions = {}
    for img_name in tqdm(os.listdir(test_images_path)):
        prediction = {'predictions': []}

        # Reading image
        img_path = os.path.join(test_images_path, img_name)
        # img = Image.open(img_path)
        img = cv2.imread(img_path)
        original_size = img.size

        contours = segm_predictor(img)

        for contour in contours:
            if contour is not None:
                crop = crop_img_by_polygon(img, contour)

                #                 plt.imshow(crop)
                #                 plt.show()

                crop = Image.fromarray(crop)
                crop = crop.resize((config.width, config.height))
                crop = np.transpose(crop, (2, 0, 1))  # H, W, C -> C, H, W
                crop = np.expand_dims(crop, 0)  # add batch dimension
                crop = torch.as_tensor(crop)
                crop = crop.cuda()

                # Prediction
                with torch.no_grad():
                    # Letter-by-letter evaluation
                    cnn_output = model.backbone(crop.float())
                    cnn_output = cnn_output.flatten(2).permute(2, 0, 1)
                    memory = model.transformer.encoder(model.pos_encoder(cnn_output).permute(1, 0, 2))

                    prob_values = 1
                    out_indexes = [text2idx_dict['SOS'], ]
                    for x in range(100):
                        target_tensor = torch.LongTensor(out_indexes).unsqueeze(0).to(config.device)

                        output = model.decoder(target_tensor)
                        output = model.pos_decoder(output)
                        output = model.transformer.decoder(output, memory)
                        output = model.out(output)

                        output_token = torch.argmax(output, dim=2)[:, -1].item()

                        prob_values = prob_values * torch.sigmoid(output[-1, 0, output_token]).item()

                        out_indexes.append(output_token)

                        if output_token == text2idx_dict['EOS']:
                            break

                    # Decode indexes
                    vec_func = lambda x: idx2text_dict[x]
                    idx2text_func = np.vectorize(vec_func)
                    outputs_ocr = np.expand_dims(idx2text_func(out_indexes), 0)

                    # Ensure that we have EOS token in the back
                    outputs_ocr = np.append(outputs_ocr, [['EOS']], axis=1)
                    # Acquire strings
                    outputs_ocr = np.apply_along_axis(lambda x: ''.join(x.tolist()[1:x.tolist().index('EOS')]), 1,
                                                      outputs_ocr)
                #                 print(' '.join(outputs_ocr))

                prediction['predictions'].append(
                    {
                        'polygon': [[int(i[0][0]), int(i[0][1])] for i in contour],
                        'text': ' '.join(outputs_ocr)
                    }
                )

        f_predictions[img_name] = prediction

    with open(output_path+'predictions.json', "w") as f:
        json.dump(f_predictions, f)

    return f_predictions


if __name__ == '__main__':
    test_images_path, output_path, visualize, path_to_font  = sys.argv[1:]

    test_pr = main(config, test_images_path, output_path)

    for img_name in tqdm(list(test_pr.keys())):
        get_image_visualization(test_pr, img_name, output_path, path_to_font, visualize)



