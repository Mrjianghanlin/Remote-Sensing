import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from model.Hrnets.hrnet import HRnet
from model.models.nets1.deeplabv3_plus_PSA import DeepLab_PSA
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.semi import SemiDataset
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log



def test(model, loader, cfg, save_path):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixels = 0
    total_pixels = 0
    class_correct_pixels = [0] * cfg['nclass']
    class_total_pixels = [0] * cfg['nclass']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            pred = model(img).argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            intersection_meter.update(intersection)
            union_meter.update(union)
            # filename = valloader.dataset.ids[0] + '.tif'
            filename = id[0].split('/')[-1]
            save_file = os.path.join(save_path, filename)
            pred_img = pred.cpu().numpy()[0]
            pred_img = np.uint8(pred_img)
            pred_img = np.squeeze(pred_img)
            Image.fromarray(pred_img).save(save_file)

            # calculate accuracy
            correct_pixels += (pred == mask.cuda()).sum().item()
            total_pixels += mask.numel()
            for i in range(cfg['nclass']):
                class_correct_pixels[i] += ((pred == i) & (mask.cuda() == i)).sum().item()
                class_total_pixels[i] += (mask == i).sum().item()

    accuracy = correct_pixels / total_pixels
    class_accuracy = [
        class_correct_pixels[i] / class_total_pixels[i] if class_total_pixels[i] > 0 else 0
        for i in range(cfg['nclass'])
    ]
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = np.mean(iou_class)
    return mIoU, iou_class, accuracy, class_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script for semantic segmentation model')
    parser.add_argument('--data-root', type=str, default='VOCdevkit')
    # parser.add_argument('--model-path', type=str, default='pretrained/best_epoch_weights1.pth')
    parser.add_argument('--model-path', type=str, default=r'C:\Users\wang\Desktop\data\image1\group\best_epoch_weights_val_mIoU_60.1482.pth')
    parser.add_argument('--save-path', type=str, default='predicted_images_city_2')

    args = parser.parse_args()

    # Set up the dataset and data loader
    valset = SemiDataset('val', args.data_root, 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    # print(id[0])
    print(valloader.dataset.ids[0])
    cfg = {
        'dataset': 'cityscapes1',
        'nclass': 19,
        'crop_size': 801,
        'data_root': 'VOCdevkit',
        'epochs': 240,
        'batch_size': 2,
        'pretrained': True,
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

    # model = DeepV3Plus(in_channels=3, n_classes=6, backbone="resnet50", pretrained=True)
    # model = DeepLabV3Plus(cfg)
    # model = DeepLab_PSA(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    # model = HRnet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'])
    # model = DeepLab(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])
    # model = PSPNet(num_classes=cfg['nclass'], backbone=cfg['backbone'], pretrained=cfg['pretrained'], downsample_factor=cfg['downsample_factor'])

    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    mIoU, iou_class, accuracy, class_accuracy = test(model, valloader,  cfg,args.save_path)
    # mIOU, iou_class = test(model, valloader, cfg)

    print('Mean IoU:', mIoU)
    print('Class-wise IoU:', iou_class)
    print('Class-wise class_accuracy:', class_accuracy)
    print(f"Mean Accuracy: {accuracy:.4f}")
