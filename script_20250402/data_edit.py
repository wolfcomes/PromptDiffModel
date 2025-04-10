import numpy as np

# 文件路径
train_npz = '../data/zinc_npz/train.npz'
test_npz = '../data/zinc_npz/test.npz'
val_npz = '../data/zinc_npz/val.npz'

# 加载数据
train_data = np.load(train_npz)
test_data = np.load(test_npz)
val_data = np.load(val_npz)

# 定义一个函数来替换数据
def replace_data(data):

    # 替换 'pocket_coord' 中的数据
    if 'pocket_coord' in data:
        pocket_coord = data['pocket_coord']
        # 将所有行的 'pocket_coord' 替换为 [0.0, 0.0, 0.0]
        pocket_coord[:] = [0.0, 0.0, 0.0]
        # 将修改后的数组存回数据中
        data['pocket_coord'] = pocket_coord

    return data

# 对 train、test、val 数据进行替换
train_data = replace_data(train_data)
test_data = replace_data(test_data)
val_data = replace_data(val_data)

# 将修改后的数据保存回原文件
np.savez(train_npz, **train_data)
np.savez(test_npz, **test_data)
np.savez(val_npz, **val_data)

print("数据替换并保存完成！")