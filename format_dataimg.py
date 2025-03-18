import os
import glob

# 设置目录路径
folder_path = 'D:\ArcGIS_data\labels'

# 获取目录中所有.tfw, .xml, 和 .ovr 文件
files_to_remove = glob.glob(os.path.join(folder_path, '*.tfw')) \
                 + glob.glob(os.path.join(folder_path, '*.xml')) \
                 + glob.glob(os.path.join(folder_path, '*.cpg')) \
                + glob.glob(os.path.join(folder_path, '*.dbf')) \
                 + glob.glob(os.path.join(folder_path, '*.ovr'))

# 删除这些文件
for file_path in files_to_remove:
    os.remove(file_path)
    print(f"Deleted {file_path}")

print("Completed deleting specified files, .tif files are kept.")
