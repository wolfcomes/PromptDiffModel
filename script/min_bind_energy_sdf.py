import os
from pathlib import Path

# 设置工作目录路径
basedir = Path('../data/docking_results/group_1')

# 用于存储所有样本的列表
all_samples = []

# 遍历 docking_results 文件夹中的子文件夹
for subdir in os.listdir(basedir):
    subdir_path = basedir / subdir
    if os.path.isdir(subdir_path):  # 确保是一个文件夹

        # 获取所有 docked_poses.pdb 文件
        docked_poses_files = list(subdir_path.glob('*_docked_poses.pdb'))

        # 遍历每一个 docked_poses.pdb 文件
        for docked_file in docked_poses_files:
            best_score = float('inf')
            best_pdb_file = None
            best_score_line = None
            best_pdb_content = []

            # 读取每个 docked_poses.pdb 文件，寻找分数最小的对接
            with open(docked_file, 'r') as f:
                current_pdb_content = []  # 用来暂存当前分子所有行
                model_started = False  # 标记当前模型是否开始
                current_score = None  # 当前模型的分数

                for line in f:
                    # 如果遇到新的 MODEL 行，开始记录一个新的分子
                    if line.startswith("MODEL"):
                        if model_started and current_score == best_score:
                            best_pdb_content = current_pdb_content  # 记录当前分子的内容

                        current_pdb_content = [line]  # 初始化当前模型的内容
                        model_started = True  # 标记模型开始

                    elif line.startswith("ENDMDL"):
                        current_pdb_content.append(line)  # 记录模型结束

                        if model_started and current_score < best_score:
                            best_score = current_score  # 更新最小分数
                            best_pdb_file = docked_file  # 更新最佳 pdb 文件
                            best_score_line = line  # 保存分数行

                        model_started = False  # 标记当前模型结束
                        current_score = None  # 清空当前模型的分数

                    # 读取分数信息：REMARK minimizedAffinity
                    if line.startswith("REMARK minimizedAffinity"):
                        current_score = float(line.split()[2])  # 提取分数

                        if current_score < best_score:
                            best_score = current_score  # 更新最小分数
                            best_pdb_file = docked_file  # 更新最佳 pdb 文件
                            best_score_line = line  # 保存分数行
                            best_pdb_content = current_pdb_content  # 保存该分子的所有内容

                    elif model_started:
                        current_pdb_content.append(line)  # 记录当前模型的其他行

            # 如果找到了最佳分数对应的分子
            if best_score_line:
                # 将分数最小的分子的 PDB 内容写入新文件
                minimized_pdb_filename = docked_file.stem + "_minimized.pdb"
                minimized_pdb_file = subdir_path / minimized_pdb_filename
                with open(minimized_pdb_file, 'w') as f_out:
                    f_out.writelines(best_pdb_content)

                # 使用 Open Babel 将 pdb 转换为 sdf 格式
                new_sdf_filename = minimized_pdb_filename.replace(".pdb", ".sdf")
                new_sdf_file = subdir_path / new_sdf_filename
                os.system(f'obabel {minimized_pdb_file} -O {new_sdf_file}')
                os.remove(minimized_pdb_file)

                # 生成新的样本数据，将 pdb 和生成的 sdf 文件存入 samples 列表
                sample = (os.path.join(subdir, minimized_pdb_filename), os.path.join(subdir, new_sdf_file.name))
                all_samples.append(sample)

# 输出所有样本
for sample in all_samples:
    print(f"PDB file: {sample[0]}, SDF file: {sample[1]}")