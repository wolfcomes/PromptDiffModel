import os
import subprocess
import shutil
import pandas as pd
from tqdm import tqdm  # Progress bar
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    filename='../data/docking_results/docking_process.log',  # 日志文件位置
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
)

# 1. 使用 smina 进行配体-受体对接并生成多个构象
def run_docking(ligand_sdf, receptor_pdb, ref_ligand_sdf, output_group_dir, file_prefix, num_poses=20):
    """ 使用 smina 进行对接，生成多个对接构象 """
    docked_dir = os.path.join(output_group_dir, file_prefix)
    docking_output = os.path.join(docked_dir, 'docked_poses.pdb')
    logging.info(f"docking输出路径: {docking_output}")
    os.makedirs(docked_dir, exist_ok=True)
    
    smina_command = [
        'smina',
        '--receptor', receptor_pdb,
        '--ligand', ligand_sdf,
        '--num_modes', str(num_poses),
        '--autobox_ligand', ref_ligand_sdf,  # 使用参考配体的坐标来定义对接盒
        '--out', docking_output
    ]
    
    subprocess.run(smina_command, check=True)
    logging.info(f"对接命令执行成功: {smina_command}")

    # 复制参考配体和受体文件到相应目录
    shutil.copy(ref_ligand_sdf, os.path.join(docked_dir, os.path.basename(ref_ligand_sdf)))
    shutil.copy(receptor_pdb, os.path.join(docked_dir, os.path.basename(receptor_pdb)))
    
    logging.info(f"参考配体和受体文件已复制：{ref_ligand_sdf}, {receptor_pdb}")

    return docking_output

# 2. 保存对接的构象
def organize_poses(docking_output, output_group_dir, ligand_file, file_prefix):
    """ 将对接的构象保存到输出目录 """
    # 创建 Docked 数据集目录
    docked_dir = os.path.join(output_group_dir, file_prefix)
    os.makedirs(docked_dir, exist_ok=True)
    
    # 将对接构象移到相应的目录
    ligand_name = os.path.basename(ligand_file).replace('.sdf', '')
    docked_output_file = os.path.join(docked_dir, f'{ligand_name}_docked_poses.pdb')
    shutil.move(docking_output, docked_output_file)
    
    logging.info(f"对接结果已保存：{docked_output_file}")

    

    return docked_output_file

# 3. 批量运行对接任务（多线程）
def batch_docking(csv_file, reference_group_dir, opt_group_dir, output_group_dir, num_poses=20, max_threads=16):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建线程池
    with ThreadPoolExecutor(max_threads) as executor:
        futures = []  # 用于存储任务的 future 对象
        
        # 使用 tqdm 添加进度条
        with tqdm(total=len(df), desc="Processing Ligands", unit="ligand") as pbar:
            # 处理每一行数据
            for idx, row in df.iterrows():
                sdf_file_name = row['SDF File']
                file_prefix = '_'.join(sdf_file_name.split('_')[:8])  # 获取前8个部分作为前缀
                reference_dir = os.path.join(reference_group_dir, file_prefix)
                opt_dir = os.path.join(opt_group_dir, file_prefix)

                # 动态获取参考配体和受体的路径
                ref_ligand_sdf = find_file_by_prefix(reference_dir, file_prefix, '.sdf')
                receptor_pdb = find_file_by_prefix(reference_dir, file_prefix, '.pdb')

                if not ref_ligand_sdf or not receptor_pdb:
                    logging.warning(f"未找到参考配体或受体文件: {file_prefix}")
                    print(f"未找到参考配体或受体文件: {file_prefix}")
                    continue

                logging.info(f"参考配体路径: {ref_ligand_sdf}, 参考受体路径: {receptor_pdb}")
                
                # 在生成组文件夹中找到所有以当前前缀为开头的文件
                ligand_files = [f for f in os.listdir(opt_dir) if f.startswith(file_prefix) and f.endswith('.sdf')]
                
                for ligand_file in ligand_files:
                    ligand_sdf_path = os.path.join(opt_dir, ligand_file)
                    logging.info(f"配体文件路径: {ligand_sdf_path}")
                    
                    # 提交每个配体的对接任务到线程池
                    futures.append(executor.submit(process_ligand, ligand_sdf_path, receptor_pdb, ref_ligand_sdf, output_group_dir, file_prefix, num_poses, pbar))

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()  # 如果任务有异常，会在此抛出
                except Exception as e:
                    logging.error(f"对接任务失败: {e}")

# 处理单个配体的任务
def process_ligand(ligand_sdf_path, receptor_pdb, ref_ligand_sdf, output_group_dir, file_prefix, num_poses, pbar):
    try:
        # 运行对接任务
        docking_output = run_docking(ligand_sdf_path, receptor_pdb, ref_ligand_sdf, output_group_dir, file_prefix, num_poses)
        # 组织并保存对接结果
        docked_dir = organize_poses(docking_output, output_group_dir, ligand_sdf_path, file_prefix)
        # 更新进度条
        pbar.update(1)
    except Exception as e:
        logging.error(f"配体对接失败: {ligand_sdf_path}, 错误: {e}")

# 4. 根据文件前缀查找文件
def find_file_by_prefix(directory, prefix, extension):
    """根据文件前缀和扩展名查找文件"""
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(extension)]
    if files:
        return os.path.join(directory, files[0])  # 返回找到的第一个文件路径
    else:
        return None

# 示例使用
if __name__ == "__main__":
    # 定义 CSV 和参考目录
    csv_file = "../data/generate_data/PPB_data_all.csv"  # 替换为你的实际 CSV 文件路径
    reference_group_dir = "../data/crossdocked_groups/group_1"  # 替换为你的参考 SDF 目录
    opt_group_dir = "../data/generate_groups/group_1"
    output_group_dir = '../data/docking_results/group_1'  # 输出目录
    num_poses = 20  # 每个配体-受体对生成的最大对接构象数
    # 创建输出目录
    os.makedirs(output_group_dir, exist_ok=True)

    # 批量对接
    batch_docking(csv_file, reference_group_dir, opt_group_dir, output_group_dir, num_poses)
    print("所有步骤已完成！")
