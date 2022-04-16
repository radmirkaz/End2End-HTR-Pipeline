import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]

        return x


class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__()

        self.config = config

        # Get backbone
        if self.config.model.backbone_name.startswith('/timm/'): 
            self.backbone = timm.create_model(self.config.model.backbone_name[6:], pretrained=self.config.model.backbone_pretrained)
        else:
            raise ValueError

        # Replace last layer with Conv2d for feature extraction
        last_layer = list(self.backbone._modules)[-1]
        setattr(self.backbone, last_layer, nn.Conv2d(in_channels=getattr(self.backbone, last_layer).in_features,
                                               out_channels=self.config.model.hidden, kernel_size=1))
        #setattr(self.backbone, last_layer, nn.Identity())
        self.backbone.global_pool = nn.Identity()

        self.pos_encoder = PositionalEncoding(self.config.model.hidden, self.config.model.dropout)
        self.pos_decoder = PositionalEncoding(self.config.model.hidden, self.config.model.dropout)
        self.decoder = nn.Embedding(self.config.model.number_of_letters, self.config.model.hidden)
        self.transformer = nn.Transformer(**self.config.model.transformer_params)

        self.out = nn.Linear(self.config.model.hidden, self.config.model.number_of_letters)

        self.x_mask = None
        self.target_mask = None
        self.memory_mask = None

    def make_len_mask(self, x):
        # Generate a ByteTensor mask of shape [N, S] or [N, T]

        return (x == 0).transpose(0, 1)

    def generate_square_subsequent_mask(self, size):
        # Generate a mask of shape [T, T] for a target with shape [N, T, E]

        mask = torch.triu(torch.ones(size, size), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))

        return mask

    def forward(self, x, target):
        # x - [N, C, W, H], target - [N, T]
        if self.target_mask is None or self.target_mask.size(0) != target.shape[1]:
            self.target_mask = self.generate_square_subsequent_mask(target.shape[1]).to(self.config.training.device) # [T, T] 

        # image
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)
        x = x.flatten(2).permute(2, 0, 1) # [S, N, E]

        x_pad_mask = self.make_len_mask(x[:, :, 0]) # [N, S]
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2) # [N, S, E]

        # target
        target_pad_mask = self.make_len_mask(target.transpose(0, 1)) # [N, T]
        target = self.decoder(target) # [N, T, E]
        target = self.pos_decoder(target) 

        # output
        output = self.transformer(x, target, src_mask=self.x_mask, tgt_mask=self.target_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=x_pad_mask, tgt_key_padding_mask=target_pad_mask, memory_key_padding_mask=x_pad_mask) # [N, T, E]
        output = self.out(output) # [N, T, N_LETTERS]

        output = output.permute(0, 2, 1) # [N, N_LETTERS, T]
        # print(output.shape)

        return output