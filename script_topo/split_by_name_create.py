import os
import random
import torch
from pathlib import Path

# 设置你的数据根目录
basedir = Path('../data/docking_results/group_2/')
all_samples = []

# 遍历所有的子目录，获取每个子目录下的 .sdf 和 .pdb 文件
for subdir in os.listdir(basedir):
    subdir_path = basedir / subdir
    if os.path.isdir(subdir_path):  # 确保是一个文件夹
        
        # 获取 subdir 下的 .sdf 和 .pdb 文件
        sdf_files = list(subdir_path.glob('*.sdf'))
        pdb_files = list(subdir_path.glob('*_pocket10.pdb'))

        # 如果当前子目录下同时包含 .sdf 和 .pdb 文件，将它们作为一个样本对
        if sdf_files and pdb_files:
            # 只保留文件名，不保留路径
            sample_pair = []
            for sdf_file in sdf_files:
                sample = (os.path.join(subdir,pdb_files[0].name), os.path.join(subdir, sdf_file.name))
                sample_pair.append(sample)
            all_samples.append(sample_pair)

# 打乱样本顺序
random.shuffle(all_samples)

# 定义数据集划分比例
train_size = 0.8
val_size = 0.1
test_size = 0.1

# 计算每个集的样本数量
n_train = int(len(all_samples) * train_size)
n_val = int(len(all_samples) * val_size)
n_test = len(all_samples) - n_train - n_val  # 剩下的作为测试集

# 划分数据集
train_samples = all_samples[:n_train]
val_samples = all_samples[n_train:n_train + n_val]
test_samples = all_samples[n_train + n_val:]

# 创建一个字典保存数据集划分信息
data_split = {
    'train': train_samples,
    'val': val_samples,
    'test': test_samples
}

# 保存为 .pt 文件
split_path = Path('../data/docking_results') / 'split_by_name_2.pt'
torch.save(data_split, split_path)

print(f"Data split saved to {split_path}")