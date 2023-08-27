import argparse
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import numpy as np
import yaml
from utils.utils import init_log, set_random_seed, load_train_model, build_model_amp, \
    AverageMeter, intersectionAndUnion, get_rank, synchronize, build_model
from data.build import get_supervised_loader
from utils.lr_helper import get_optimizer, get_scheduler
from utils.loss_helper import get_criterion
import os

import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

parser = argparse.ArgumentParser(description='Supervised Semantic Segmentaion')
parser.add_argument("--config", type=str, default="/6THardDisk/zhanweiqi/semi-cons/config/cityscape/full.yaml")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--local_rank", type=int, default=0)

logger = init_log(name='global')
logger.propagate = 0


# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


def main():
    global args, cfg, scaler
    scaler = GradScaler()
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cudnn.enabled = True
    cudnn.benchmark = True

    num_gpus = int(os.environ['WORLD_SIZE']
                   ) if 'WORLD_SIZE' in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        synchronize()

    rank = get_rank()
    if rank == 0:
        logger.info(cfg)
    if args.seed is not None:
        print('set random seed to ', args.seed)
        set_random_seed(args.seed)
    if not os.path.exists(cfg['saver']['snapshot_dir']):
        os.makedirs(cfg['saver']['snapshot_dir'])
    model = build_model(cfg)
    device = torch.device('cuda')
    model.to(device)
    print(f"是否分布式：{distributed}")
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ["LOCAL_RANK"])], output_device=int(os.environ["LOCAL_RANK"]),
            find_unused_parameters=True
        )
    print("初始化模型")
    if cfg['saver']['pretrain']:
        state_dict = torch.load(cfg['saver']['pretrain'], map_location='cpu')[
            'model_state']
        print('Load trained model from ', str(cfg['saver']['pretrain']))
        model = model.load_state_dict(state_dict)

    # model = DataParallel(model).cuda()
    # model = model.cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    criterion = get_criterion(cfg)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg['dataset']['ignore_index'])

    train_loader, val_loader = get_supervised_loader(cfg, seed, dis=distributed)

    cfg_trainer = cfg['trainer']
    cfg_optim = cfg_trainer['optimizer']

    params_list = []
    # modules_back = model.module.encoder
    # # modules_back = model.module.backbone
    # modules_head = model.module.decoder
    # params_list.append(dict(params=modules_back.parameters(),
    # lr=cfg_optim['kwargs']['lr']))
    # params_list.append(dict(params=modules_head.parameters(),
    # lr=cfg_optim['kwargs']['lr'] * 10))
    # modules_back = model.module.backbone
    # modules_head = model.module.classifier
    params_list.append(dict(params=[param for name, param in model.named_parameters()
                                    if 'backbone' in name], lr=cfg_optim['kwargs']['lr']))
    params_list.append(dict(params=[param for name, param in model.named_parameters()
                                    if 'backbone' not in name], lr=cfg_optim['kwargs']['lr'] * 10))
    optimizer = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(cfg_trainer, len(train_loader), optimizer)

    best_prec = 0
    if rank == 0:
        logger.info(f"there are {train_loader.__len__()} images in labeled loader")

    torch.cuda.empty_cache()

    for epoch in range(cfg_trainer['epochs']):
        # print("进入训练循环")
        t_start = time.time()
        train(model, optimizer, lr_scheduler, criterion, train_loader, epoch)
        if cfg_trainer['eval_on']:
            prec = validate(model, val_loader, epoch, criterion)
            if rank == 0:
                if prec > best_prec:
                    best_prec = prec
                    state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state':
                        optimizer.state_dict()}
                    torch.save(state, os.path.join(
                        cfg['saver']['snapshot_dir'], f"best_{cfg['saver']['snapshot_name']}_" + str(seed) + '.pth'))
                    logger.info(f"Currently, the best val resut is: {best_prec}")

        # save the last epoch's checkpoint
        if (epoch == (cfg_trainer['epochs'] - 1) or epoch == (cfg_trainer['epochs'] - 2)) and rank == 0:
            state = {'epoch': epoch, 'model_state': model.state_dict(
            ), 'optimizer_state': optimizer.state_dict()}
            torch.save(state, os.path.join(
                cfg['saver']['snapshot_dir'],
                f"last_{cfg['saver']['snapshot_name']}_epoch" + str(epoch) + '_' + str(seed) + '.pth'))
            logger.info(f"Save Checkpoint {epoch}")
        t_end = time.time()
        if rank == 0:
            print('time for one epoch', t_end - t_start)


def train(model, optimizer, lr_scheduler, criterion, data_loader, epoch):
    model.train()
    # data_loader_iter = iter(data_loader)
    data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)
    num_classes, ignore_index = cfg['dataset']['classes'], cfg['dataset']['ignore_index']
    rank = get_rank()
    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    for step in range(len(data_loader)):
        # print("我在训练")
        i_iter = epoch * len(data_loader) + step
        lr = lr_scheduler.get_lr()
        lr_scheduler.step()

        images, masks = data_loader_iter.next()
        images = images.cuda()
        masks = masks.long().cuda()
        # print(images[0])
        # print(masks.max())
        # print(masks.min())

        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        # print(loss)
        optimizer.step()
        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.data.argmax(1).cpu().numpy()
        # print(outputs.unique())
        # print(np.unique(outputs))
        # print(masks.unique())
        masks = masks.cpu().numpy()

        intersection, union, target = intersectionAndUnion(
            outputs, masks, num_classes, ignore_index)
        # print(f'intersection = {intersection}')
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()
        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)
        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        if i_iter % 50 == 0 and rank == 0:
            iou_class = intersection_meter.sum / (union_meter.sum + 1e-15)
            accuracy_class = intersection_meter.sum / \
                             (target_meter.sum + 1e-15)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)

            logger.info(
                f"iter = {i_iter} of {cfg['trainer']['epochs'] * len(data_loader)} completed, LR = {lr} loss = {losses.avg} mIoU = {mIoU} mAcc = {mAcc}")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    if rank == 0:
        logger.info(
            f'=============epoch[{epoch}]=============, Train mIoU = {mIoU} mAcc = {mAcc}')


def validate(model, data_loader, epoch, criterion):
    model.eval()
    rank = get_rank()
    num_classes, ignore_index = cfg['dataset']['classes'], cfg['dataset']['ignore_index']

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    losses = AverageMeter()

    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.cuda()
        labels = labels.long().cuda()
        with autocast():
            with torch.no_grad():
                preds = model(images)
                loss = criterion(preds, labels)
        preds = F.softmax(preds, dim=1)
        output = preds.data.argmax(1).cpu().numpy()
        target = labels.cpu().numpy()

        intersection, union, target = intersectionAndUnion(
            output, target, num_classes, ignore_index)

        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()
        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)
        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())
        reduced_loss = loss.clone()
        dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-15)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-15)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    if rank == 0:
        print('=======mIoU:=======', mIoU)
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
        logger.info(
            f'=============epoch[{epoch}]=============, Validation loss = {losses.avg} mIoU = {mIoU} mAcc = {mAcc}')

    torch.save(mIoU, 'eval_metric.pth.tar')
    return mIoU


if __name__ == '__main__':
    main()
