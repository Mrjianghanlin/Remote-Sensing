import os
import torch

from model.semseg.deeplabv3plus import DeepLabV3Plus


def main():
    cfg = {
        'dataset': 'pascal',
        'data_root': 'VOCdevkit/VOCdevkit/VOC2012',
        'nclass': 6,
        'crop_size': 512,
        'backbone': 'resnet50',
        'pretrained':False,
        'epochs': 300,
        'batch_size': 2,
        'lr': 0.001,
        'lr_multi': 1.0,
        'criterion': {
            'name': 'CELoss',
            'kwargs': {
                'ignore_index': 255
            }
        },
        'conf_thresh': 0.95,
        'model': 'deeplabv3plus',
        'backbone': 'resnet50',
        'replace_stride_with_dilation': [False, False, True],
        'dilations': [6, 12, 18]
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = r"C:\Users\wang\Desktop\UniMatch-main\UniMatch-main\exp_supervised\deeplabv3plus_r50\5%\best.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # # option1
    # net = resnet34()
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # # change fc layer structure
    # in_channel = net.fc.in_features
    # net.fc = nn.Linear(in_channel, 5)

    # option2

    net = DeepLabV3Plus(cfg)
    state_dict = net.state_dict()#查看权重的类别,指向具体发键值对
    #
    pre_weights = torch.load(model_weight_path, map_location=device)
    # pre_weights = torch.load("./model_data/swin_base_patch4_window12_384.pth", map_location="cpu")["model"]
    # model.load_state_dict(weights_dict, strict=False)
    del_key = []
    for k in list(pre_weights.keys()):
        if "head" in k:
            del pre_weights[k]

    for key in del_key:
        del pre_weights[key]

    missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
