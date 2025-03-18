from torchstat import stat
from model.models.nets1.deeplabv3_plus import DeepLab
cfg = {
        'dataset': 'pascal',
        'data_root': 'VOCdevkit/VOCdevkit_all/VOC2007',
        'nclass': 6,
        'crop_size': 512,
        'pretrained': True,
        'epochs': 200,
        'batch_size': 10,
        'lr': 0.001,
        'lr_multi': 10.0,
        'criterion': {
            'name': 'CELoss',
            'kwargs': {
                'ignore_index': 255
            }
        },
        'conf_thresh': 0.95,
        # 'backbone': 'resnset101',
        # 'backbone': 'mobilenet',
        'backbone': 'hrnet',
        # 'backbone': 'hrnetv2_w32',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18],
        'model_path': "logs/best.pth",
        'num_workers': 4,
        'downsample_factor': 16,
        'in_channels': 3,
    }
# 创建模型实例
# model = DeepLab(num_classes=3, backbone="hrnet", downsample_factor=16, pretrained=True)
from model.models.nets1.deeplabv3_plus_PSA import DeepLab_PSA
from model.semseg.deeplabv3plus import DeepLabV3Plus

# model = DeepLabV3Plus(cfg)
model = DeepLab_PSA(num_classes=3, backbone="hrnet", downsample_factor=16, pretrained=False)

# 使用torchstat来计算GFLOPs
stat(model, (3, 512, 512))
