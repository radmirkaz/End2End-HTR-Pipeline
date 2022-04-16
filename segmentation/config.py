from omegaconf import OmegaConf, DictConfig, ListConfig

config = {
    'general': {
        'seed': 10
    },
    'paths': {
        'path_to_annotations_json': 'data/annotations.json',
        'path_to_images': 'data/images',

        'path_to_active_training_data': 'active_training_data',
        'path_to_checkpoints': 'checkpoints'
    }
}

config = OmegaConf.create(config)