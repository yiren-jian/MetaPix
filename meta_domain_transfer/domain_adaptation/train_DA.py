import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from meta_domain_transfer.model.unet import get_unet, get_unet2, get_unet3, get_unet4
from meta_domain_transfer.utils.func import adjust_learning_rate
from meta_domain_transfer.utils.func import loss_calc, bce_loss
from meta_domain_transfer.utils.loss import entropy_loss, robust_entropy_loss
from meta_domain_transfer.utils.func import prob_2_entropy
from meta_domain_transfer.utils.viz_segmask import colorize_mask

from collections import OrderedDict


def loopy(dataloader):
    while True:
        for x in iter(dataloader): yield x


def save_model(model, i_iter, cfg):
    # save model in compatible way
    params = dict()
    params_trained = model.state_dict().copy()

    params['conv1_1.weight'] = params_trained['conv1x2_1.weight']
    params['conv1_1.bias']   = params_trained['conv1x2_1.bias']
    params['conv1_2.weight'] = params_trained['conv1x2_2.weight']
    params['conv1_2.bias']   = params_trained['conv1x2_2.bias']

    params['conv2_1.weight'] = params_trained['conv2_1.weight']
    params['conv2_1.bias']   = params_trained['conv2_1.bias']
    params['conv2_2.weight'] = params_trained['conv2_2.weight']
    params['conv2_2.bias']   = params_trained['conv2_2.bias']
    params['conv3_1.weight'] = params_trained['conv3_1.weight']
    params['conv3_1.bias']   = params_trained['conv3_1.bias']
    params['conv3_2.weight'] = params_trained['conv3_2.weight']
    params['conv3_2.bias']   = params_trained['conv3_2.bias']
    params['conv3_3.weight'] = params_trained['conv3_3.weight']
    params['conv3_3.bias']   = params_trained['conv3_3.bias']
    params['conv4_1.weight'] = params_trained['conv4_1.weight']
    params['conv4_1.bias']   = params_trained['conv4_1.bias']
    params['conv4_2.weight'] = params_trained['conv4_2.weight']
    params['conv4_2.bias']   = params_trained['conv4_2.bias']
    params['conv4_3.weight'] = params_trained['conv4_3.weight']
    params['conv4_3.bias']   = params_trained['conv4_3.bias']
    params['conv5_1.weight'] = params_trained['conv5_1.weight']
    params['conv5_1.bias']   = params_trained['conv5_1.bias']
    params['conv5_2.weight'] = params_trained['conv5_2.weight']
    params['conv5_2.bias']   = params_trained['conv5_2.bias']
    params['conv5_3.weight'] = params_trained['conv5_3.weight']
    params['conv5_3.bias']   = params_trained['conv5_3.bias']
    params['fc6.weight']     = params_trained['fc6.weight']
    params['fc6.bias']       = params_trained['fc6.bias']
    params['fc7.weight']     = params_trained['fc7.weight']
    params['fc7.bias']       = params_trained['fc7.bias']
    params['score.weight']   = params_trained['score.weight']
    params['score.bias']     = params_trained['score.bias']

    params['score_pool3.weight']   = params_trained['score_pool3.weight']
    params['score_pool3.bias']     = params_trained['score_pool3.bias']
    params['score_pool4.weight']   = params_trained['score_pool4.weight']
    params['score_pool4.bias']     = params_trained['score_pool4.bias']

    print('taking snapshot ...')
    print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
    torch.save(params,
               osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def draw_weight_in_tensorboard(writer, images, i_iter, weights):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image', grid_image, i_iter)

    output_w = weights.cpu().data[0]
    grid_image = make_grid(output_w, 3, normalize=False, range=(0, 1))
    writer.add_image(f'PixelWeight', grid_image, i_iter)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def train_target_only(model, targetloader, cfg):
    ''' CityScapes training with "labels" (Target Only)
    '''
    # Create the model and start the training.
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        model.adjust_learning_rate(optimizer, i_iter, cfg)

        # supervised training with cross entropy loss (on target cityscapes)
        _, batch = targetloader_iter.__next__()
        image, label, _, _ = batch
        _, pred = model(None, image.to(device))

        loss_seg = loss_calc(pred, label, None, device)
        loss_seg.backward()
        optimizer.step()

        current_losses = {'loss_seg': loss_seg}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_in_tensorboard(writer, image, i_iter, pred, num_classes, 'Target')


def train_direct_joint(model, trainloader, targetloader, cfg):
    ''' Domain Transfer Learning with Direct Joint
    '''
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model = nn.DataParallel(model)
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.module.get_parameters(bias=False)},
                {'params': model.module.get_parameters(bias=True),
                 'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.module.get_parameters(bias=False)},
                {'params': model.module.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)


    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP)):

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        model.module.adjust_learning_rate(optimizer, i_iter, cfg)

        # UDA Training
        # train on source
        _, src_batch = trainloader_iter.__next__()
        image_src, label_src, _, _ = src_batch
        # train on target
        _, trg_batch = targetloader_iter.__next__()
        image_trg, label_trg, _, _ = trg_batch

        _, pred_src = model(image_src.to(device), None)
        loss_seg_src = loss_calc(pred_src, label_src, None, device)
        loss_seg_src.backward()

        _, pred_trg = model(None, image_trg.to(device))
        loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)
        loss_seg_trg.backward()

        optimizer.step()

        current_losses = {'loss_seg_trg': loss_seg_trg}
        current_losses.update({'loss_seg_src': loss_seg_src})
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.module.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)


def train_scale_source_loss(model, trainloader, targetloader, cfg):

    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                 'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)


    trainloader_iter = loopy(trainloader)
    targetloader_iter = loopy(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)):
        _train_scale_source_loss_step(trainloader_iter, targetloader_iter, i_iter,
                                    model, optimizer, viz_tensorboard, writer, device, cfg)


def _train_scale_source_loss_step(trainloader_iter, targetloader_iter, i_iter,
                                model, optimizer, viz_tensorboard, writer, device, cfg):

    optimizer.zero_grad()
    # adapt LR if needed
    model.adjust_learning_rate(optimizer, i_iter, cfg)

    src_batch = next(trainloader_iter)
    image_src, label_src, _, _ = src_batch

    trg_batch = next(targetloader_iter)
    image_trg, label_trg, _, _ = trg_batch

    # train on source
    _, pred_src = model(image_src.to(device), None)
    loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))
    loss_seg_src *= cfg.TRAIN.SCALE_LOSS

    loss_seg_src = loss_seg_src.mean()
    loss_seg_src.backward()
    current_losses = {'loss_seg_src': loss_seg_src.item()}

    # train on target
    _, pred_trg = model(None, image_trg.to(device))
    loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)
    loss_seg_trg.backward()

    current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
    print_losses(current_losses, i_iter)

    # update the paramters in the segnet
    optimizer.step()

    if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
        print('taking snapshot ...')
        print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
        torch.save(model.state_dict(),
                   osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
        if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
            pass
    sys.stdout.flush()

    # Visualize with tensorboard
    if viz_tensorboard:
        log_losses_tensorboard(writer, current_losses, i_iter)

        if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
            pass


def train_meta_pixel_weight(model, trainloader, targetloader, cfg):

    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # AUXILIARY NETWORK: Pixel Weight
    model_W = get_unet2(input_channel=23, num_classes=2)
    # model_W = get_unet4(input_channel=39, num_classes=2)
    model_W.train()
    model_W.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                 'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # W's optimizer
    if cfg.TRAIN.META_OPTIMIZER == 'Adam':
        optimizer_W = optim.Adam(
                model_W.parameters(),
                lr=cfg.TRAIN.META_LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)
    elif cfg.TRAIN.META_OPTIMIZER == 'SGD':
        optimizer_W = optim.SGD(
                model_W.parameters(),
                lr=cfg.TRAIN.META_LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)


    trainloader_iter = loopy(trainloader)
    targetloader_iter = loopy(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)):
        _train_meta_pixel_weight_step1(trainloader_iter, targetloader_iter, i_iter,
                                       model, model_W, optimizer, optimizer_W,
                                       viz_tensorboard, writer, device, cfg)
        if i_iter % cfg.TRAIN.META_ALTERNATION_RATE == 0:
            _train_meta_pixel_weight_step2(trainloader_iter, targetloader_iter, i_iter,
                                           model, model_W, optimizer, optimizer_W,
                                           viz_tensorboard, writer, device, cfg)


def _train_meta_pixel_weight_step1(trainloader_iter, targetloader_iter, i_iter,
                                   model, model_W, optimizer, optimizer_W,
                                   viz_tensorboard, writer, device, cfg):

    optimizer.zero_grad()
    optimizer_W.zero_grad()
    # adapt LR if needed
    model.adjust_learning_rate(optimizer, i_iter, cfg)

    src_batch = next(trainloader_iter)
    image_src, label_src, _, _ = src_batch
    label_one_hot_src = F.one_hot(label_src.clamp(0,19).long(), num_classes=20)
    label_one_hot_src = label_one_hot_src.transpose(2, 3).transpose(1, 2).contiguous()

    trg_batch = next(targetloader_iter)
    image_trg, label_trg, _, _ = trg_batch
    label_one_hot_trg = F.one_hot(label_trg.clamp(0,19).long(), num_classes=20)
    label_one_hot_trg = label_one_hot_trg.transpose(2, 3).transpose(1, 2).contiguous()

    # train on source
    _, pred = model(image_src.to(device), image_trg.to(device))
    pred_src = pred[0].unsqueeze(0)
    pred_trg = pred[1].unsqueeze(0)

    with torch.no_grad():
        model_W_output = model_W(torch.cat((image_src, label_one_hot_src.float()), dim=1).to(device))
        # pred_src_copy = pred_src.data.clone().detach()
        # model_W_output = model_W(torch.cat((pred_src_copy, label_one_hot_src.float().to(device)), dim=1))

    pixel_weight = model_W_output[:,1,:,:]
    pixel_weight = pixel_weight.data.clone().detach().squeeze(1)
    loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))

    loss_seg_src *= pixel_weight
    pixel_weight_mean = pixel_weight.mean().data
    current_losses = {'pixel_weight_mean': pixel_weight_mean}

    loss_seg_src = loss_seg_src.mean()
    loss_seg_src.backward(retain_graph=True)
    current_losses.update({'loss_seg_src': loss_seg_src.item()})

    # train on target
    loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)
    loss_seg_trg.backward()

    current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
    print_losses(current_losses, i_iter)

    # update the paramters in the segnet
    optimizer.step()

    if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
        print('taking snapshot ...')
        print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
        torch.save(model.state_dict(),
                   osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
        if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
            pass
    sys.stdout.flush()

    # Visualize with tensorboard
    if viz_tensorboard:
        log_losses_tensorboard(writer, current_losses, i_iter)

        if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
            draw_weight_in_tensorboard(writer, image_src, i_iter, pixel_weight)


def _train_meta_pixel_weight_step2(trainloader_iter, targetloader_iter, i_iter,
                                   model, model_W, optimizer, optimizer_W,
                                   viz_tensorboard, writer, device, cfg):

    # reset optimizers
    optimizer.zero_grad()
    optimizer_W.zero_grad()
    vgg_lr = optimizer.param_groups[0]['lr']

    src_batch = next(trainloader_iter)
    image_src, label_src, _, _ = src_batch
    label_one_hot_src = F.one_hot(label_src.clamp(0,19).long(), num_classes=20)
    label_one_hot_src = label_one_hot_src.transpose(2, 3).transpose(1, 2).contiguous()

    trg_batch = next(targetloader_iter)
    image_trg, label_trg, _, _ = trg_batch

    _, pred_src = model(image_src.to(device), None)

    model_W_output = model_W(torch.cat((image_src, label_one_hot_src.float()), dim=1).to(device))
    # pred_src_copy = pred_src.data.clone().detach()
    # model_W_output = model_W(torch.cat((pred_src_copy, label_one_hot_src.float().to(device)), dim=1))

    pixel_weight = model_W_output[:,-1,:,:]

    # current theta
    fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

    loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))
    loss_seg_src *= pixel_weight
    current_losses = {'pixel_weight_mean': pixel_weight.mean().data}

    loss_seg_src = loss_seg_src.mean()
    grads = torch.autograd.grad(loss_seg_src, model.parameters(), create_graph=True, allow_unused=True)

    # compute theta^+ by applying sgd on main loss
    fast_new_weights = OrderedDict()
    for  ((name, param), grad ) in zip(fast_weights.items(), grads):
        if grad is not None:
            fast_new_weights[name] = param - vgg_lr * grad
        else:
            fast_new_weights[name] = param

    # compute target loss with the updated thetat^+
    _, pred_trg = model.forward(None, image_trg.to(device), fast_new_weights)
    loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)

    current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
    print_losses(current_losses, i_iter)

    loss_seg_trg.backward()
    optimizer_W.step()

    sys.stdout.flush()


def train_W_with_FCN(model, trainloader, targetloader, cfg):
    saved_state_dict = torch.load('/home/yiren/META_DOMAIN_TRANSFER/pretrained_models/model_gen0.pth')
    model.load_state_dict(saved_state_dict)

    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # AUXILIARY NETWORK: Pixel Weight
    model_W = get_unet2(input_channel=23, num_classes=2)
    # model_W = get_unet4(input_channel=39, num_classes=2)
    model_W.train()
    model_W.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                 'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # W's optimizer
    if cfg.TRAIN.META_OPTIMIZER == 'Adam':
        optimizer_W = optim.Adam(
                model_W.parameters(),
                lr=cfg.TRAIN.META_LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)
    elif cfg.TRAIN.META_OPTIMIZER == 'SGD':
        optimizer_W = optim.SGD(
                model_W.parameters(),
                lr=cfg.TRAIN.META_LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)


    trainloader_iter = loopy(trainloader)
    targetloader_iter = loopy(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)):
        # reset optimizers
        optimizer.zero_grad()
        optimizer_W.zero_grad()
        vgg_lr = optimizer.param_groups[0]['lr']

        src_batch = next(trainloader_iter)
        image_src, label_src, _, _ = src_batch
        label_one_hot_src = F.one_hot(label_src.clamp(0,19).long(), num_classes=20)
        label_one_hot_src = label_one_hot_src.transpose(2, 3).transpose(1, 2).contiguous()

        trg_batch = next(targetloader_iter)
        image_trg, label_trg, _, _ = trg_batch

        _, pred_src = model(image_src.to(device), None)

        model_W_output = model_W(torch.cat((image_src, label_one_hot_src.float()), dim=1).to(device))
        # pred_src_copy = pred_src.data.clone().detach()
        # model_W_output = model_W(torch.cat((pred_src_copy, label_one_hot_src.float().to(device)), dim=1))

        pixel_weight = model_W_output[:,-1,:,:]

        # current theta
        fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

        loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))
        loss_seg_src *= pixel_weight
        current_losses = {'pixel_weight_mean': pixel_weight.mean().data}

        loss_seg_src = loss_seg_src.mean()
        grads = torch.autograd.grad(loss_seg_src, model.parameters(), create_graph=True, allow_unused=True)

        # compute theta^+ by applying sgd on main loss
        fast_new_weights = OrderedDict()
        for  ((name, param), grad ) in zip(fast_weights.items(), grads):
            if grad is not None:
                fast_new_weights[name] = param - vgg_lr * grad
            else:
                fast_new_weights[name] = param

        # compute target loss with the updated thetat^+
        _, pred_trg = model.forward(None, image_trg.to(device), fast_new_weights)
        loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)

        current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
        print_losses(current_losses, i_iter)

        loss_seg_trg.backward()
        optimizer_W.step()

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model_W.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_W_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                pass
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_weight_in_tensorboard(writer, image_src, i_iter, pixel_weight)


def train_FCN_with_W(model, trainloader, targetloader, cfg):
    saved_state_dict = torch.load('/home/yiren/META_DOMAIN_TRANSFER/pretrained_models/model_gen0.pth')
    model.load_state_dict(saved_state_dict)

    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # AUXILIARY NETWORK: Pixel Weight
    model_W = get_unet2(input_channel=23, num_classes=2)
    # model_W = get_unet4(input_channel=39, num_classes=2)
    model_W.load_state_dict(torch.load('/home/yiren/META_DOMAIN_TRANSFER/pretrained_models/W_gen0.pth'))
    model_W.train()
    model_W.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                 'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # W's optimizer
    if cfg.TRAIN.META_OPTIMIZER == 'Adam':
        optimizer_W = optim.Adam(
                model_W.parameters(),
                lr=cfg.TRAIN.META_LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)
    elif cfg.TRAIN.META_OPTIMIZER == 'SGD':
        optimizer_W = optim.SGD(
                model_W.parameters(),
                lr=cfg.TRAIN.META_LEARNING_RATE,
                betas=(0.9, 0.999),
                weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)


    trainloader_iter = loopy(trainloader)
    targetloader_iter = loopy(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)):
        optimizer.zero_grad()
        optimizer_W.zero_grad()
        # adapt LR if needed
        model.adjust_learning_rate(optimizer, i_iter, cfg)

        src_batch = next(trainloader_iter)
        image_src, label_src, _, _ = src_batch
        label_one_hot_src = F.one_hot(label_src.clamp(0,19).long(), num_classes=20)
        label_one_hot_src = label_one_hot_src.transpose(2, 3).transpose(1, 2).contiguous()

        trg_batch = next(targetloader_iter)
        image_trg, label_trg, _, _ = trg_batch
        label_one_hot_trg = F.one_hot(label_trg.clamp(0,19).long(), num_classes=20)
        label_one_hot_trg = label_one_hot_trg.transpose(2, 3).transpose(1, 2).contiguous()

        # train on source
        _, pred = model(image_src.to(device), image_trg.to(device))
        pred_src = pred[0].unsqueeze(0)
        pred_trg = pred[1].unsqueeze(0)

        with torch.no_grad():
            model_W_output = model_W(torch.cat((image_src, label_one_hot_src.float()), dim=1).to(device))
            # pred_src_copy = pred_src.data.clone().detach()
            # model_W_output = model_W(torch.cat((pred_src_copy, label_one_hot_src.float().to(device)), dim=1))

        pixel_weight = model_W_output[:,1,:,:]
        pixel_weight = pixel_weight.data.clone().detach().squeeze(1)
        loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))

        loss_seg_src *= pixel_weight
        pixel_weight_mean = pixel_weight.mean().data
        current_losses = {'pixel_weight_mean': pixel_weight_mean}

        loss_seg_src = loss_seg_src.mean()
        loss_seg_src.backward(retain_graph=True)
        current_losses.update({'loss_seg_src': loss_seg_src.item()})

        # train on target
        loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)
        loss_seg_trg.backward()

        current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
        print_losses(current_losses, i_iter)

        # update the paramters in the segnet
        optimizer.step()

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(),
                       osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                pass
        sys.stdout.flush()

        # Visualize with tensorboard
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)

            if i_iter % cfg.TRAIN.TENSORBOARD_VIZRATE == cfg.TRAIN.TENSORBOARD_VIZRATE - 1:
                draw_weight_in_tensorboard(writer, image_src, i_iter, pixel_weight)


def train_meta_image_weight(model, trainloader, targetloader, cfg):

    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # AUXILIARY NETWORK: Image Weight
    model_W = get_unet4(input_channel=23, num_classes=2)
    # model_W = get_unet4(input_channel=39, num_classes=2)
    model_W.train()
    model_W.to(device)

    # OPTIMIZERS
    # segnet's optimizer
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                 'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(
            [
                {'params': model.get_parameters(bias=False)},
                {'params': model.get_parameters(bias=True),
                'lr': cfg.TRAIN.LEARNING_RATE * 2}
            ],
            lr=cfg.TRAIN.LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # W's optimizer
    optimizer_W = optim.Adam(
            model_W.parameters(),
            lr=cfg.TRAIN.META_LEARNING_RATE,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.META_WEIGHT_DECAY)


    trainloader_iter = loopy(trainloader)
    targetloader_iter = loopy(targetloader)
    for i_iter in tqdm(range(cfg.TRAIN.MAX_ITERS)):
        _train_meta_image_weight_step1(trainloader_iter, targetloader_iter, i_iter,
                                       model, model_W, optimizer, optimizer_W,
                                       viz_tensorboard, writer, device, cfg)
        _train_meta_image_weight_step2(trainloader_iter, targetloader_iter, i_iter,
                                       model, model_W, optimizer, optimizer_W,
                                       viz_tensorboard, writer, device, cfg)


def _train_meta_image_weight_step1(trainloader_iter, targetloader_iter, i_iter,
                                   model, model_W, optimizer, optimizer_W,
                                   viz_tensorboard, writer, device, cfg):

    optimizer.zero_grad()
    optimizer_W.zero_grad()
    # adapt LR if needed
    model.adjust_learning_rate(optimizer, i_iter, cfg)

    src_batch = next(trainloader_iter)
    image_src, label_src, _, _ = src_batch
    label_one_hot_src = F.one_hot(label_src.clamp(0,19).long(), num_classes=20)
    label_one_hot_src = label_one_hot_src.transpose(2, 3).transpose(1, 2).contiguous()

    trg_batch = next(targetloader_iter)
    image_trg, label_trg, _, _ = trg_batch
    label_one_hot_trg = F.one_hot(label_trg.clamp(0,19).long(), num_classes=20)
    label_one_hot_trg = label_one_hot_trg.transpose(2, 3).transpose(1, 2).contiguous()

    # train on source
    _, pred = model(image_src.to(device), image_trg.to(device))
    pred_src = pred[0].unsqueeze(0)
    pred_trg = pred[1].unsqueeze(0)
    with torch.no_grad():
        model_W_output = model_W(torch.cat((image_src, label_one_hot_src.float()), dim=1).to(device))
        # pred_src_copy = pred_src.data.clone().detach()
        # model_W_output = model_W(torch.cat((pred_src_copy, label_one_hot_src.float().to(device)), dim=1))
        image_weight = model_W_output[-1]

    image_weight = image_weight.data.clone().detach()
    loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))

    loss_seg_src *= image_weight
    image_weight = image_weight.data
    current_losses = {'image_weight': image_weight}

    loss_seg_src = loss_seg_src.mean()
    loss_seg_src.backward(retain_graph=True)
    current_losses.update({'loss_seg_src': loss_seg_src.item()})

    # train on target
    loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)
    loss_seg_trg.backward()

    current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
    print_losses(current_losses, i_iter)

    # update the paramters in the segnet
    optimizer.step()

    if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
        print('taking snapshot ...')
        print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
        torch.save(model.state_dict(),
                   osp.join(cfg.TRAIN.SNAPSHOT_DIR, f'model_{i_iter}.pth'))
        if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
            pass
    sys.stdout.flush()

    # Visualize with tensorboard
    if viz_tensorboard:
        log_losses_tensorboard(writer, current_losses, i_iter)



def _train_meta_image_weight_step2(trainloader_iter, targetloader_iter, i_iter,
                                   model, model_W, optimizer, optimizer_W,
                                   viz_tensorboard, writer, device, cfg):

    # reset optimizers
    optimizer.zero_grad()
    optimizer_W.zero_grad()
    vgg_lr = optimizer.param_groups[0]['lr']

    src_batch = next(trainloader_iter)
    image_src, label_src, _, _ = src_batch
    label_one_hot_src = F.one_hot(label_src.clamp(0,19).long(), num_classes=20)
    label_one_hot_src = label_one_hot_src.transpose(2, 3).transpose(1, 2).contiguous()

    trg_batch = next(targetloader_iter)
    image_trg, label_trg, _, _ = trg_batch

    _, pred_src = model(image_src.to(device), None)

    model_W_output = model_W(torch.cat((image_src, label_one_hot_src.float()), dim=1).to(device))
    # pred_src_copy = pred_src.data.clone().detach()
    # model_W_output = model_W(torch.cat((pred_src_copy, label_one_hot_src.float().to(device)), dim=1))
    image_weight = model_W_output[-1]

    # current theta
    fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

    # pixel_weight = pixel_weight.squeeze(1)
    loss_seg_src = torch.nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=255)(pred_src, label_src.long().to(device))

    loss_seg_src *= image_weight
    current_losses = {'image_weight': image_weight.data}

    loss_seg_src = loss_seg_src.mean()
    grads = torch.autograd.grad(loss_seg_src, model.parameters(), create_graph=True, allow_unused=True)

    # compute theta^+ by applying sgd on main loss
    fast_new_weights = OrderedDict()
    for  ((name, param), grad ) in zip(fast_weights.items(), grads):
        if grad is not None:
            fast_new_weights[name] = param - vgg_lr * grad
        else:
            fast_new_weights[name] = param

    # compute target loss with the updated thetat^+
    _, pred_trg = model.forward(None, image_trg.to(device), fast_new_weights)
    loss_seg_trg = loss_calc(pred_trg, label_trg, None, device)

    current_losses.update({'loss_seg_trg': loss_seg_trg.item()})
    print_losses(current_losses, i_iter)

    loss_seg_trg.backward()
    optimizer_W.step()

    sys.stdout.flush()



def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'TargetOnly':
        train_target_only(model, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'DirectJoint':
        train_direct_joint(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'MetaPixelWeight':
        train_meta_pixel_weight(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'train_W_with_FCN':
        train_W_with_FCN(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'train_FCN_with_W':
        train_FCN_with_W(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'MetaImageWeight':
        train_meta_image_weight(model, trainloader, targetloader, cfg)
    elif cfg.TRAIN.DA_METHOD == 'ScaleSourceLoss':
        train_scale_source_loss(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")
