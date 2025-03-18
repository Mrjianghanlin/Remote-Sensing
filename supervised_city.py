import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.semi import SemiDataset
from model.Hrnets.hrnet import HRnet
from model.models.nets1.deeplabv3_plus import DeepLab
from model.models.nets1.deeplabv3_plus_PSA import DeepLab_PSA
import os; os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str, default="splits/cityscapes/c_1_1/labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="splits/cityscapes/c_1_1/unlabeled.txt")
parser.add_argument('--save-path', type=str, default="exp/cityscapes/supervised/c_1_1/pre")
parser.add_argument('--freeze-epochs', type=int, default=10)


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:

            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter.update(intersection)
            union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = {
        'dataset': 'cityscapes',
        'nclass': 35,
        'crop_size': 801,
        'data_root': 'VOCdevkit/VOCdevkitgt5/VOC2007',
        'epochs': 240,
        'batch_size': 2,
        'pretrained':True,
        'lr': 0.005,
        'lr_multi': 1.0,
        'criterion': {
            'name': 'OHEM',
            'kwargs': {
                'ignore_index': 255,
                'thresh': 0.7,
                'min_kept': 200000
            }
        },
        'conf_thresh': 0,
        'backbone': 'hrnet',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18]
    }

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    writer = SummaryWriter(args.save_path)

    os.makedirs(args.save_path, exist_ok=True)

    # model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    model = DeepLab_PSA(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    # model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    # model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])


    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    model = model.cuda()

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss.item(), iters)

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))

        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }
        torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch_{epoch}_{total_loss.val:.4f}.pth"))

        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_epoch_weights.pth"))
            # torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            best_checkpoint_filename = os.path.join(args.save_path, f"best_epoch_weights_val_mIoU_{mIoU:.4f}.pth")
            torch.save(model.state_dict(), best_checkpoint_filename)


if __name__ == '__main__':
    main()
