import os
import numpy as np
from PIL import Image
import glob

# Cityscapes的19个类别的颜色映射
cityscapes_palette = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

def colorize_mask(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id, color in enumerate(cityscapes_palette):
        color_mask[mask == cls_id] = color
    return color_mask

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_file in glob.glob(input_folder + '/*.png'):
        label_image = Image.open(img_file)
        label_image = np.array(label_image)

        colored_mask = colorize_mask(label_image)
        colored_mask_image = Image.fromarray(colored_mask)

        # 保存上色后的图像到输出文件夹
        base_name = os.path.basename(img_file)
        colored_mask_image.save(os.path.join(output_folder, base_name))

# 定义输入和输出文件夹
input_folder = r'C:\Users\wang\Desktop\data\label1'
output_folder = r'C:\Users\wang\Desktop\data\1'

# 处理整个文件夹
process_folder(input_folder, output_folder)
