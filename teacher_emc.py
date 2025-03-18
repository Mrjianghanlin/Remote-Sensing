
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
from tqdm import tqdm



from dataset.semi import SemiDataset
from model.Hrnets.hrnet import HRnet
from model.models.model import HRNet
# from model.models.nets.deeplabv3_plus import DeepLab
from model.models.model.res_unet_dbb import Res_Unet
from model.models.nets1.deeplabv3_plus import DeepLab
from model.models.nets1.deeplabv3_plus_cff import DeepLab_cff
from model.models.nets1.deeplabv3_plus_sum import DeepLab_Dense
from model.pspnets.pspnet import PSPNet
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.models.nets1.deeplabv3_training import weights_init

from model.semseg.unet import UNet
from model.unets.unet import Unet
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log




parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--labeled-id-path', type=str, default="splits/pascal/un_1_1/labeled.txt")
parser.add_argument('--unlabeled-id-path', type=str, default="splits/pascal/un_1_1/unlabeled.txt")
parser.add_argument('--save-path', type=str, default="exp/dali/c_1_1/hrnetv2_w48/ewc")
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--freeze-epochs', type=int, default=0)

class EWC:
    def __init__(self, model):
        self.model = model
        self.saved_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        self.fisher_diagonal = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}

    def compute_fisher(self, data_loader, criterion):
        self.model.train()
        for img, mask in data_loader:
            img, mask = img.cuda(), mask.cuda()
            self.model.zero_grad()
            output = self.model(img)
            loss = criterion(output, mask)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher_diagonal[n] += (p.grad ** 2).detach() / len(data_loader)

    def ewc_loss(self, lamda=1e-4):
        loss = 0
        for n, p in self.model.named_parameters():
            loss += (self.fisher_diagonal[n] * (p - self.saved_params[n]) ** 2).sum()
        return lamda * loss.cuda()



def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    total=0
    correct=0
    best_acc = 0  # Initialize the best accuracy as 0
    with torch.no_grad():
        loader = tqdm(loader, desc="Evaluating")
        for i, (img, mask, id) in enumerate(loader):
            img = img.cuda()

            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                # final = torch.zeros(b, 19, h, w).cuda()
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
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
            total += img.size(0) * img.size(2) * img.size(3)
            correct += (pred == mask.cuda()).sum().item()
    if total == 0:
        acc = 0
    else:
        acc = correct / total
    if acc > best_acc:
        best_acc = acc
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    print(f"Mean Accuracy: {acc:.4f}")
    print(f"Best Accuracy: {best_acc:.4f}")  # Print the best accuracy

    return mIOU, iou_class

def distillation_loss(student_output, teacher_output, temperature=1.0):
    # 计算知识蒸馏损失
    teacher_probs = torch.nn.functional.softmax(teacher_output / temperature, dim=1)
    loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_output / temperature, dim=1), teacher_probs)
    return temperature * temperature * loss
def main():
    args = parser.parse_args()
    # model_path = "./pretrained/best_epoch_weights.pth"
    model_path=''


    cfg = {
        'dataset': 'pascal',
        'data_root': 'VOCdevkit/VOCdevkit_all/VOC2007',
        'nclass': 35,
        'crop_size': 256,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 2,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {
            'name': 'CELoss',
            'kwargs': {
                'ignore_index': 255
            }
        },
        'conf_thresh': 0.95,
        # 'backbone': 'resnet101_ibn_a',
        # 'backbone': 'xception',
        # 'backbone': 'mobilenet',
        # 'backbone': 'hrnet',
        'backbone': 'hrnetv2_w48',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18],
        'model_path': "logs/best.pth",
        'num_workers': 0,
        'downsample_factor': 16,
        'in_channels': 3,
    }
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # Remove distributed training related code

    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    writer = SummaryWriter(args.save_path)

    os.makedirs(args.save_path, exist_ok=True)



    # print(model)

    # model = DeepLabV3Plus(cfg)
    # model = Unet(num_classes=cfg['nclass'], backbone=cfg['backbone'],pretrained=cfg['pretrained'])
    # model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'],pretrained=cfg['pretrained'])
    # model=DeepLab_Dense(num_classes=cfg['nclass'], backbone=cfg['backbone'],pretrained=cfg['pretrained'])
    # model=DeepLab_Dense(num_classes=cfg['nclass'], backbone=cfg['backbone'],pretrained=cfg['pretrained'])

    # model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])
    # model = PSPNet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])
    student_model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'],
                                  pretrained=cfg['pretrained']).cuda()
    teacher_model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'],
                                  pretrained=cfg['pretrained']).cuda()
    # teacher_model.load_state_dict(torch.load(model_path))
    teacher_model.eval()



    local_rank =0
    # if not cfg['pretrained']:
    #     weights_init(model)
    #
    # if model_path != '':
    #     # ------------------------------------------------------#
    #     #   权值文件请看README，百度网盘下载
    #     # ------------------------------------------------------#
    #     if local_rank == 0:
    #         print('Load weights {}.'.format(model_path))
    #
    #     # ------------------------------------------------------#
    #     #   根据预训练权重的Key和模型的Key进行加载
    #     # ------------------------------------------------------#
    #     model_dict = model.state_dict()
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     pretrained_dict = torch.load(model_path, map_location=device)
    #     load_key, no_load_key, temp_dict = [], [], {}
    #     for k, v in pretrained_dict.items():
    #         if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #             temp_dict[k] = v
    #             load_key.append(k)
    #         else:
    #             no_load_key.append(k)
    #     model_dict.update(temp_dict)
    #     model.load_state_dict(model_dict)
    #     # ------------------------------------------------------#
    #     #   显示没有匹配上的Key
    #     # ------------------------------------------------------#
    #     if local_rank == 0:
    #         print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    #         print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    #         print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#

    # if cfg['Freeze_Train']:
    #     for name, param in model.backbone.named_parameters():
    #         if 'layer1' in name:
    #             param.requires_grad = False
    #
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             print(f'Frozen parameter: {name}')
    logger.info('Total params: {:.1f}M\n'.format(count_params(student_model)))

    optimizer = SGD(student_model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    model = student_model.cuda()

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=cfg['num_workers'], drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=cfg['num_workers'], drop_last=False)

    ewc = EWC(student_model)
    ewc.compute_fisher(trainloader, criterion)
    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(cfg['model_path'])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']

        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))
        if epoch < args.freeze_epochs:
            for name, param in model.backbone.named_parameters():
                if 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name:
                # if 'layer1' in name or 'layer2' in name or 'layer3' in name :
                # if 'layer1' in name or 'layer2' in name:
                # if 'layer1' in name  :
                    param.requires_grad = False
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(f'Frozen parameter: {name}')
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(f'Frozen parameter: {name}')

        model.train()
        total_loss = AverageMeter()

        for i, (img, mask) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()

            student_pred = student_model(img)
            teacher_pred = teacher_model(img).detach()  # 确保不计算梯度

            # 原始损失
            ce_loss = criterion(student_pred, mask)

            # 蒸馏损失
            distill_loss = distillation_loss(student_pred, teacher_pred)

            # 合并损失
            ewc_loss_value = ewc.ewc_loss()
            combined_loss = ce_loss + distill_loss + ewc_loss_value

            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()

            total_loss.update(ce_loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            writer.add_scalar('train/loss_all', ce_loss.item(), iters)
            writer.add_scalar('train/loss_x', ce_loss.item(), iters)

            if i % (max(2, len(trainloader) // 8)) == 0:
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}'.format(
                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
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

        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_epoch_weights.pth"))
            # torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            best_checkpoint_filename = os.path.join(args.save_path, f"best_epoch_weights_val_mIoU_{mIoU:.4f}.pth")
            torch.save(model.state_dict(), best_checkpoint_filename)
        torch.save(model.state_dict(), os.path.join(args.save_path, f"model_epoch_{epoch}_{total_loss.val:.4f}.pth"))


if __name__ == '__main__':
    main()
