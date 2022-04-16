from omegaconf import OmegaConf, DictConfig, ListConfig

config = {
    'general': {
        'experiment_name': 'default',
        'seed': 2000,
        'num_classes': 1,
    },
    'paths': {
        'path_to_images': 'data/nto_finals_data/images/',
        # 'path_to_inference_images': 'data/test',

        'path_to_csv': 'data/nto_finals_data/data_ext1.csv',

        'path_to_external_images': 'data/img/',
        'path_to_external_images2': 'data/third_dataset/train/',
        'path_to_external_images3': 'data/forms/images/',
        'path_to_external_images4': 'data/letters/letters/',

        'path_to_checkpoints': './submission/checkpoints/',
        'path_to_sample_submission': 'data/sample_submission.csv',
        'path_to_predicted_submission': 'submission.csv',

        'log_name': 'log.txt',
        'path_to_weights': '',

    },
    'training': {
        'num_epochs': 50 ,
        'lr': 0.0001, # target learning rate = base lr * warmup_multiplier if warmup_multiplier > 1.0 (0.00001)
        'mixed_precision': True,
        'gradient_accumulation': False,
        'gradient_accumulation_steps': 4,
        'early_stopping_epochs': 30,

        'prediction_threshold': 0.5,

        'warmup_scheduler': False,
        'warmup_epochs': 3,
        'warmup_multiplier': 100, # lr needs to be divided by warmup_multiplier if warmup_multiplier > 1.0

        'debug': False,
        'debug_number_of_samples': 20000,

        'device': 'cuda',
        'improvement_tracking_mode': 'metric', # metric or val_loss
        'save_step': 9,
        'verbose_plots': True,
    },
    'data': {
        'id_column': 'id',
        'target_columns': ['label'], # mask column, not used as masks come from an npz file
        'image_format': '', # not used, "id" column contains image format

        'train_batch_size': 128,
        'val_batch_size': 1,
        'test_batch_size': 1,
        'num_workers': 8,

        'kfold': {
            'use_kfold': True,
            'name': 'StratifiedKFold',
            'split_on_column': 'length',
            'group_column': None, # used only for GroupKFold (str or None)
            'current_fold': 0, # 0 - 4
            'train_all_folds': False,
            'params': {
                'n_splits': 5,
                'shuffle': True,
                'random_state': '${general.seed}'
            }
        },

        # Sizes of data splits
        'train_size': 0.8,
        'val_size': 0.2,
        'test_size': 0.0,
        
        # Progeressive resizing (not used if sizes are equal)
        'start_size': 224, 
        'final_size': 224,
        'step_size': 32,

        # Training augmentations (not implemented)
        'cutmix': False,
        'cutmix_alpha': 1.0,

        'mixup': False,
        'mixup_alpha': 1.0,

        'fmix': False
    },
    'augmentations': {
        'pre_transforms': [
            # {
            #     'name': 'Normalize',
            #     'params': {
            #         'mean': (0.485, 0.456, 0.406),
            #         'std': (0.229, 0.224, 0.225),
            #         'p': 1.0
            #     }
            # },
            {       
                'name': 'Resize',
                'params': {
                    'height': 128,
                    'width': 384,
                    'p': 1.0
                }
            },
        ],
        'transforms': [
            {
                'name': 'Rotate',
                'params': {
                    'limit': [-5, 5],
                    'p': 0.5
                }
            },
            {
                'name': '/custom/ExtraLinesAugmentation',
                'params': {
                    "number_of_lines": 4,
                    "width_of_lines": 5,
                    "p": 0.5
                }
            },
            # {
            #     'name': 'Blur',
            #     'params': {
            #         'blur_limit': 10,
            #         'p': 0.3
            #     }
            # },

        ],
        'post_transforms': [
            {
                'name': '/custom/to_tensorv2',
                'params': {
                    'p': 1.0
                }
            }
        ]
    },
    'model': {
        'name': '/custom/Seq2SeqModel',

        'freeze_batchnorms': False, # not used
        'norm_no_decay': False, # not used

        'backbone_name': '/timm/swsl_resnext101_32x4d',
        'backbone_pretrained': True,
        'hidden': 128,
        'dropout': 0.1,

        'number_of_letters': 152, # output dimension for a model

        'transformer_params': {
            'd_model': '${model.hidden}',
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 1,
            'dim_feedforward': 128 * 4, # hidden * 4
            'dropout': '${model.dropout}',
            'activation': 'relu',
            'batch_first': True
        },

        'checkpoint_path': 'best.pt',
    },
    'optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.lr}',
            #'weight_decay': 0.001
        }
    },
    'scheduler': {
        'name': 'CosineAnnealingLR', 
        'interval': 'epoch', # epoch or step
        'params': {
            'T_max': 7,
            'eta_min': 1e-7
        }
    },
    'loss': {
        'name': 'CrossEntropyLoss',
        'params': {
            'ignore_index': 0, # index for 'PAD'
            'label_smoothing': 0.1
        },
    },
    'metric': {
        'name': '/custom/cer',
        'mode': 'min',
        'params': {
            
        }
    },
    'inference': {
        'inference': False,
    },
    'resume_from_checkpoint': {
        # implemented
        'resume': True,
        'optimizer_state': False,
        'scheduler_state': False,
        'epochs_since_improvement': 0,
        'last_epoch': 0
    } 
}

config = OmegaConf.create(config)
