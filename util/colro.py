import numpy as np
from PIL import Image

# Cityscapes的19个类别的颜色映射
cityscapes_palette = {
    'road': (128, 64, 128),
    'sidewalk': (244, 35, 232),
    'building': (70, 70, 70),
    'wall': (102, 102, 156),
    'fence': (190, 153, 153),
    'pole': (153, 153, 153),
    'traffic light': (250, 170, 30),
    'traffic sign': (220, 220, 0),
    'vegetation': (107, 142, 35),
    'terrain': (152, 251, 152),
    'sky': (70, 130, 180),
    'person': (220, 20, 60),
    'rider': (255, 0, 0),
    'car': (0, 0, 142),
    'truck': (0, 0, 70),
    'bus': (0, 60, 100),
    'train': (0, 80, 100),
    'motorcycle': (0, 0, 230),
    'bicycle': (119, 11, 32)
}

def colorize_mask(mask):
    # 创建一个空白的彩色图像
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for cls_id, color in enumerate(cityscapes_palette.values()):
        color_mask[mask == cls_id] = color

    return color_mask

# 读取标签图像 (确保它是单通道，每个像素值对应一个类别)
label_image_path = r'D:\study\banjiandu\UniMatch-main\Erhai semantic segmentation\splits\cityscapes\frankfurt_000000_000294_gtFine_labelTrainIds.png'
label_image = Image.open(label_image_path)
label_image = np.array(label_image)

# 上色
colored_mask = colorize_mask(label_image)

# 将结果保存为图片
colored_mask_image = Image.fromarray(colored_mask)
colored_mask_image.save('colored_mask.png')
