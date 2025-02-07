import os
import shutil
import logging
from tqdm import tqdm  # 引入 tqdm

# 设置源目录（分组后的目录），如 group_1, group_2 等
group_dir = '../data/crossdocked_groups'
num_groups = 48  # 总组数

# 设置日志配置
logging.basicConfig(
    filename="add_subfolders.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("开始执行将文件移动到次级目录的脚本...")

# 遍历所有分组目录
for group_num in tqdm(range(2, 49), desc="处理组目录", unit="组"):
    group_folder = os.path.join(group_dir, f"group_{group_num}")

    # 确保目标组目录存在
    if not os.path.exists(group_folder):
        logging.warning(f"目录 {group_folder} 不存在，跳过...")
        continue

    logging.info(f"正在处理目录: {group_folder}...")

    # 遍历组目录中的所有文件
    files = os.listdir(group_folder)
    for file in tqdm(files, desc=f"处理中 group_{group_num} 中的文件", unit="文件", leave=False):
        file_path = os.path.join(group_folder, file)

        if os.path.isfile(file_path):
            # 提取文件前缀（假设前缀是文件名中第 9 个'_'之前的部分）
            file_prefix = '_'.join(file.split('_')[:8])

            # 为每个文件前缀创建一个次级目录
            subfolder = os.path.join(group_folder, file_prefix)

            # 如果次级目录不存在，则创建它
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
                logging.info(f"创建次级目录: {subfolder}")

            # 将文件移动到次级目录
            shutil.move(file_path, os.path.join(subfolder, file))
            logging.info(f"文件已移动: {file} -> {subfolder}")

logging.info("脚本执行完成！")
print("文件已成功移动到次级目录。")