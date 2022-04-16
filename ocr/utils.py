import numpy as np  
import pandas as pd

import os
import gc
import time
import random
import pickle
from tqdm import tqdm

import sklearn

import torch
import timm
import torchvision
import torch.nn as nn
from torch.cuda import amp

import matplotlib.pyplot as plt

from config import config
from data import get_loaders, get_loader_inference
from custom.scheduler import GradualWarmupSchedulerV2
from custom.augmentations import cutmix, mixup
from custom.models import Seq2SeqModel
from custom.metrics import cer


def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model(config):
    '''Get PyTorch model.'''

    if not config.resume_from_checkpoint.resume:
        print('Model:', config.model.name)

        if config.model.name.startswith('/custom/'):
            model = globals()[config.model.name[8:]](config, ) # add additional parameters for seq2seq model
        else:
            raise RuntimeError('Unknown model source. Use /smp/ or /custom/.')

        model = model.to(config.training.device)

        return model
    else:
        print('Model:', config.model.name)
        model_name = config.model.name
        checkpoint_path = config.model.checkpoint_path

        if model_name.startswith('/custom/'):
            model = globals()[model_name[8:]](config)
        else:
            raise RuntimeError('Unknown model source. Use /custom/.')

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.training.device))
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print(f'Loaded {model_name}')
        else:
            model.load_state_dict(checkpoint)

        model = model.to(config.training.device)

        return model


def get_optimizer(config, model):
    '''Get PyTorch optimizer'''
    checkpoint_path = config.model.checkpoint_path

    if not config.resume_from_checkpoint.optimizer_state:
        if config.optimizer.name.startswith('/custom/'):
            optimizer = globals()[config.optimizer.name[8:]](model.parameters(), **config.optimizer.params)
        else:
            optimizer = getattr(torch.optim, config.optimizer.name)(model.parameters(), **config.optimizer.params)

        return optimizer
    else:
        if config.optimizer.name.startswith('/custom/'):
            optimizer = globals()[config.optimizer.name[8:]](model.parameters(), **config.optimizer.params)
        else:
            optimizer = getattr(torch.optim, config.optimizer.name)(model.parameters(), **config.optimizer.params)

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.training.device))
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Loaded optimizer')

        return optimizer


def get_scheduler(config, optimizer):
    '''Get PyTorch scheduler'''
    checkpoint_path = config.model.checkpoint_path

    if not config.resume_from_checkpoint.scheduler_state:
        if config.scheduler.name.startswith('/custom/'):
            scheduler = globals()[config.scheduler.name[8:]](optimizer, **config.scheduler.params)
        else:
            scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)

        if config.training.warmup_scheduler:
            final_scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=config.training.warmup_multiplier,
                                                     total_epoch=config.training.warmup_epochs, after_scheduler=scheduler)

            return final_scheduler
        else:
            return scheduler
    else:
        if config.scheduler.name.startswith('/custom/'):
            scheduler = globals()[config.scheduler.name[8:]](optimizer, **config.scheduler.params)
        else:
            scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(config.training.device))
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f'Loaded scheduler')

        return scheduler

    
def get_loss(config):
    '''Get PyTorch loss function.'''

    if config.loss.name.startswith('/custom/'):
        loss = globals()[config.loss.name[8:]](**config.loss.params)
    else:
        loss = getattr(nn, config.loss.name)(**config.loss.params)

    return loss


def get_metric(config, y_true, y_pred):
    '''Calculate metric.'''
    
    predictions = y_pred

    if config.metric.name.startswith('/custom/'):
        score = globals()[config.metric.name[8:]](y_true, predictions, **config.metric.params)
    else:
        score = getattr(sklearn.metrics, config.metric.name)(y_true, predictions, **config.metric.params)
    
    return score


def train(config, model, train_loader, optimizer, scheduler, loss_function, epoch, image_size, scaler):
    '''Train loop.'''

    print('Training')

    model.train()

    if config.model.freeze_batchnorms:
        for name, child in (model.named_children()):
            if name.find('BatchNorm') != -1:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True
    
    total_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader)):
        inputs, targets = batch
        inputs, targets = inputs.to(config.training.device), targets.to(config.training.device)
        # Training Augmentations (not implemented)
        if config.data.mixup or config.data.cutmix or config.data.fmix:
            p = random.uniform(0, 1)
            if p < 0.5:
                inputs, labels = mixup(inputs, labels, alpha=config.data.mixup_alpha)

        if not config.training.gradient_accumulation:
            optimizer.zero_grad()
        
        if config.training.mixed_precision:
            with amp.autocast():
                outputs = model(inputs.float(), targets[:, :-1])

                if config.data.mixup or config.data.cutmix or config.data.fmix:
                    if p < 0.5:
                        loss = loss_function(outputs, targets['target']) * targets['lam'] + loss_function(outputs, targets['shuffled_target']) * (1.0 - targets['lam'])
                    else:
                        loss = loss_function(outputs, targets)
                else:
                    # print(outputs.shape, targets[:, 1:].shape)
                    loss = loss_function(outputs, targets[:, 1:])
                    # print(loss)

                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
        else:
            outputs = model(inputs.float())
            
            if config.data.mixup or config.data.cutmix or config.data.fmix:
                if p < 0.5:
                    loss = loss_function(outputs, targets['target']) * targets['lam'] + loss_function(outputs, targets['shuffled_target']) * (1.0 - targets['lam'])
                else:
                    loss = loss_function(outputs, targets)
            else:
                loss = loss_function(outputs, targets)

        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.training.gradient_accumulation:
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        elif config.training.mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if config.scheduler.interval == 'step':
            if config.training.warmup_scheduler:
                if epoch >= config.training.warmup_epochs:
                    scheduler.step()
            else:
                scheduler.step()
    
    if config.training.warmup_scheduler:
        if epoch < config.training.warmup_epochs:
            scheduler.step()
        else:
            if config.scheduler.interval == 'epoch':
                scheduler.step()
    else:
        if config.scheduler.interval == 'epoch':
            scheduler.step()

    print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(train_loader)


def validation(config, model, val_loader, loss_function):
    '''Validation loop.'''

    print('Validating')

    with open('data/idx2text_dict.pkl', 'rb') as f:
        idx2text_dict = pickle.load(f)
    with open('data/text2idx_dict.pkl', 'rb') as f:
        text2idx_dict = pickle.load(f)

    model.eval()

    total_loss = 0.0
    total_metric = 0.0

    preds, targets = [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_loader)):
            inputs, targets = batch
            inputs, targets = inputs.to(config.training.device), targets.to(config.training.device)

            # Letter-by-letter evaluation
            cnn_output = model.backbone(inputs.float())
            cnn_output = cnn_output.flatten(2).permute(2, 0, 1)
            memory = model.transformer.encoder(model.pos_encoder(cnn_output).permute(1, 0, 2))

            prob_values = 1
            out_indexes = [text2idx_dict['SOS'], ]
            for x in range(100):
                target_tensor = torch.LongTensor(out_indexes).unsqueeze(0).to(config.training.device)

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
            outputs = np.expand_dims(idx2text_func(out_indexes), 0)
            targets = idx2text_func(targets.to('cpu').numpy())

            # Ensure that we have EOS token in the back
            outputs = np.append(outputs, [['EOS']], axis=1)

            #outputs = model(inputs.float(), targets) 

            # if step == 0:
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(masks.to('cpu').numpy()[0, :, :, :].transpose(1, 2, 0))

            #     plt.subplot(1, 2, 2)
            #     plt.imshow(outputs.to('cpu').numpy()[0, :, :, :].transpose(1, 2, 0))

            #     plt.show()

            # loss = loss_function(outputs, targets)
            # total_loss += loss.item()

            # outputs = torch.argmax(outputs, dim=1).to('cpu').numpy()

            # # Decode indexes
            # vec_func = lambda x: idx2text_dict[x]
            # idx2text_func = np.vectorize(vec_func)
            # outputs = idx2text_func(outputs)
            # targets = idx2text_func(targets.to('cpu').numpy())
            
            # Acquire strings
            targets = np.apply_along_axis(lambda x: ''.join(x.tolist()[1:x.tolist().index('EOS')]), 1, targets)
            outputs = np.apply_along_axis(lambda x: ''.join(x.tolist()[1:x.tolist().index('EOS')]), 1, outputs)
            if step == 1:
                print(targets)
                print(outputs)

            metric = get_metric(config, targets, outputs)
            total_metric += metric

    return total_loss / len(val_loader), total_metric / len(val_loader)


def test(config, model, test_loader, loss_function):
    '''Testing loop.'''

    print('Testing')

    model.eval()

    total_loss = 0.0

    preds, targets = [], []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader)):
            inputs, labels = batch
            inputs, labels = inputs.to(config.training.device), labels.squeeze(1).to(config.training.device)

            outputs = model(inputs.float()) 

            preds.append(outputs.sigmoid().to('cpu').numpy())
            targets.append(labels.to('cpu').numpy())

            loss = loss_function(outputs, labels)
            total_loss += loss.item()

    metric = get_metric(config, np.concatenate(targets), np.concatenate(preds))
    print('Test Loss:', total_loss / len(test_loader), '\nTest Metric:', metric)


def run(config):
    '''Main function.'''

    # Create working directory
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints)

    if config.data.test_size == 0.0:
        train_loader, val_loader = get_loaders(config)
    else:
        train_loader, val_loader, test_loader = get_loaders(config)

    torch.cuda.empty_cache()

    # Get objects
    model = get_model(config)
    # print(model)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    loss_function = get_loss(config)

    if config.model.norm_no_decay:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': config.optimizer.params.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

    # Initializing metrics and logging
    train_losses, val_losses, metrics, learning_rates = [], [], [], []
    if config.metric.mode == 'max':
        best_metric = 0
    else:
        best_metric = np.inf
    best_val_loss = np.inf

    if config.resume_from_checkpoint.epochs_since_improvement:
        epochs_since_improvement = config.resume_from_checkpoint.epochs_since_improvement
    else:
        epochs_since_improvement = 0
    
    print('Testing ' + config.general.experiment_name + ' approach')
    if config.paths.log_name:
        with open(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), 'w') as file:
            file.write('Testing ' + config.general.experiment_name + ' approach\n')
    
    # Training
    # Store transforms to use them after warmup stage
    transforms = config.augmentations.transforms
    image_size = config.data.start_size
    
    if config.resume_from_checkpoint.last_epoch > 0:
        current_epoch = config.resume_from_checkpoint.last_epoch
    else:
        current_epoch = 0

    for epoch in range(current_epoch, config.training.num_epochs):
        print('\nEpoch: ' + str(epoch + 1))

        # Applying progressive resizing
        if image_size < config.data.final_size and epoch > config.training.warmup_epochs:
            image_size += config.data.size_step
            config.augmentations.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': image_size,
                        'width': image_size,
                        'p': 1.0
                    }
                }
            ]
        
        # No transforms for warmup stage
        if epoch < config.training.warmup_epochs:
            config.augmentations.transforms = [
                {
                    'name': 'HorizontalFlip',
                    'params': {
                        'p': 0.5
                    }
                }
            ]
        else:
            config.augmentations.transforms = transforms

        if image_size < config.data.final_size:
            print('Progressive resizing, current image size: ' + str(image_size))

        start_time = time.time()

        train_loss = train(config, model, train_loader, optimizer, scheduler, loss_function, epoch, image_size, scaler)
        val_loss, current_metric = validation(config, model, val_loader, loss_function)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics.append(current_metric)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        if config.training.improvement_tracking_mode == 'metric':
            if current_metric <= best_metric:
                print('New Record!')

                best_metric = current_metric
                epochs_since_improvement = 0
            
                save_model(config, model, epoch, train_loss, val_loss, current_metric, optimizer, epochs_since_improvement, 
                        f'best.pt', scheduler, image_size)
            else:
                epochs_since_improvement += 1
        elif config.training.improvement_tracking_mode == 'val_loss':
            if val_loss < best_val_loss:
                print('New Record!')

                best_val_loss = val_loss
                epochs_since_improvement = 0
            
                save_model(config, model, epoch, train_loss, val_loss, current_metric, optimizer, epochs_since_improvement, 
                        f'best.pt', scheduler, image_size)
            else:
                epochs_since_improvement += 1

        if epoch % config.training.save_step == 0:
            save_model(config, model, epoch, train_loss, val_loss, current_metric, optimizer, epochs_since_improvement,
                       f'{epoch + 1}_epoch.pt', scheduler, image_size)

        t = int(time.time() - start_time)

        if config.training.improvement_tracking_mode == 'metric':
            print_report(config, t, train_loss, val_loss, current_metric, best_metric)
        elif config.training.improvement_tracking_mode == 'val_loss':
            print_report(config, t, train_loss, val_loss, current_metric, best_val_loss)

        if config.paths.log_name:
            if config.training.improvement_tracking_mode == 'metric':
                save_log(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), epoch + 1,
                        train_loss, val_loss, current_metric)
            elif config.training.improvement_tracking_mode == 'val_loss':
                save_log(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), epoch + 1,
                        train_loss, val_loss, best_val_loss)

        if epochs_since_improvement == config.training.early_stopping_epochs:
            print('Training has been interrupted by early stopping.')
            break
        
        torch.cuda.empty_cache()
        gc.collect()

    if config.data.test_size > 0.0:
        test(config, model, test_loader, loss_function)

    if config.training.verbose_plots:
        draw_plots(train_losses, val_losses, metrics, learning_rates)


def save_model(config, model, epoch, train_loss, val_loss, metric, optimizer, epochs_since_improvement, name, scheduler, image_size):
    '''Save PyTorch model.'''

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'metric': metric,
        'optimizer': optimizer.state_dict(),
        'epochs_since_improvement': epochs_since_improvement,
        'scheduler': scheduler.state_dict(),
        'image_size': image_size,
    }, os.path.join(config.paths.path_to_checkpoints, name))


def draw_plots(train_losses, val_losses, metrics, lr_changes):
    '''Draw plots of losses, metrics and learning rate changes.'''

    # Learning rate changes
    plt.plot(range(len(lr_changes)), lr_changes, label='Learning Rate')
    plt.legend()
    plt.title('Learning rate changes')
    plt.show()

    # Validation and train losses
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Changes of validation and train losses')
    plt.show()

    # Metric changes
    plt.plot(range(len(metrics)), metrics, label='Metric')
    plt.legend()
    plt.title('Metric changes')
    plt.show()


def print_report(config, t, train_loss, val_loss, metric, best_metric):
    '''Print report of one epoch.'''

    print(f'Time: {t} s')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    print(f'Current Metric: {metric:.4f}')
    
    if config.training.improvement_tracking_mode == 'metric':
        print(f'Best Metric: {best_metric:.4f}')
    elif config.training.improvement_tracking_mode == 'val_loss': 
        print(f'Best Loss: {best_metric:.4f}')


def save_log(path, epoch, train_loss, val_loss, best_metric):
    '''Save log of one epoch.'''

    with open(path, 'a') as file:
        file.write('epoch: ' + str(epoch) + 'train_loss: ' + str(round(train_loss, 5)) + 
                   ' val_loss: ' + str(round(val_loss, 5)) + ' best_metric: ' + 
                   str(round(best_metric, 5)) + '\n')


def get_model_inference(config, model_name, model_params, checkpoint_path):
    '''Get PyTorch model.'''

    print('Model:', model_name)

    if model_name.startswith('/timm/'): 
        model = timm.create_model(model_name[6:], pretrained=False, in_chans=config.model.in_channels)
    elif model_name.startswith('/torch/'):
        model = getattr(torchvision.models, model_name[7:])(pretrained=False)
    elif config.model.name.startswith('/custom/'):
        model = globals()[model_name[8:]](**model_params)
    else:
        raise RuntimeError('Unknown model source. Use /timm/ or /torch/.')

    last_layer = list(model._modules)[-1]
    try:
        setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer).in_features,
                                                out_features=config.general.num_classes, bias=True))
    except torch.nn.modules.module.ModuleAttributeError:
        setattr(model, last_layer, nn.Linear(in_features=getattr(model, last_layer)[1].in_features,
                                                out_features=config.general.num_classes, bias=True))

    checkpoint = torch.load(checkpoint_path)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(config.training.device)

    return model


def inference(config):
    list_of_models = []

    for x in range(len(config.inference.list_of_models)):
        if len(config.inference.list_of_models_params[x]) == 0:
            model = get_model_inference(config, config.inference.list_of_models[x], None, config.inference.list_of_checkpoints[x])
        else:
            model = get_model_inference(config, config.inference.list_of_models[x], config.inference.list_of_models_params[x], config.inference.list_of_checkpoints[x])

        list_of_models.append(model)

    torch.cuda.empty_cache()   
    data_loader = get_loader_inference(config)

    preds = []
    with torch.no_grad():
        for img in tqdm(data_loader):
            outputs = 0
            img = img.to(config.training.device)

            for model in list_of_models:
                model.eval()
                outputs += model(img.float()) # No sigmoid for regression
            
            outputs = outputs / len(list_of_models)
            #outputs = np.argmax(outputs.to('cpu'), axis=1).tolist()
            outputs = outputs.to('cpu').detach().tolist()

            preds += outputs
    
    sample_submission = pd.read_csv(config.paths.path_to_sample_submission)
    sample_submission[config.inference.preds_columns] = preds

    # sample_submission = sample_submission.drop(['file_path'], axis=1)

    sample_submission.to_csv(config.paths.path_to_predicted_submission, index=False)



