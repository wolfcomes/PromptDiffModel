import os
import MDAnalysis as mda
import shutil
import logging

# 配置日志
logging.basicConfig(
    filename='../data/docking_results/rmsd_filter.log',  # 日志文件位置
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
)

def calculate_rmsd(reference_pdb, docked_pdb):
    """计算参考配体与对接构象之间的 RMSD"""
    # 读取参考配体和对接构象
    u_ref = mda.Universe(reference_pdb)
    u_docked = mda.Universe(docked_pdb)

    # 假设参考配体和对接构象的配体部分原子名称一致，并且长度相同
    ref_atoms = u_ref.select_atoms("resname LIG")  # 根据实际情况调整
    docked_atoms = u_docked.select_atoms("resname LIG")  # 根据实际情况调整

    # 计算 RMSD
    rmsd = ref_atoms.rmsd(docked_atoms)
    return rmsd

def filter_rmsd(docked_dir, reference_pdb, output_dir, rmsd_threshold=1.0):
    """筛选 RMSD 小于 threshold 的对接构象"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历对接结果目录
    for root, _, files in os.walk(docked_dir):
        for file in files:
            if file.endswith('.pdb'):
                docked_pdb = os.path.join(root, file)
                logging.info(f"正在计算 RMSD: {docked_pdb}")

                # 计算 RMSD
                rmsd = calculate_rmsd(reference_pdb, docked_pdb)

                if rmsd < rmsd_threshold:
                    # 保留 RMSD 小于阈值的对接构象
                    logging.info(f"RMSD 小于 {rmsd_threshold}: {docked_pdb} (RMSD = {rmsd})")
                    # 将符合条件的结果复制到输出目录
                    shutil.copy(docked_pdb, os.path.join(output_dir, file))
                else:
                    logging.info(f"RMSD 大于 {rmsd_threshold}: {docked_pdb} (RMSD = {rmsd})")
                    # 可选择删除或忽略 RMSD 超过阈值的构象
                    os.remove(docked_pdb)
                    logging.info(f"已删除对接结果: {docked_pdb}")

# 示例使用
if __name__ == "__main__":
    docked_dir = "../data/docking_results/group_1"  # 对接结果的目录
    reference_pdb = "../data/crossdocked_groups/group_1/reference.pdb"  # 参考配体的 PDB 文件
    output_dir = "../data/docking_results/filtered_poses"  # 筛选后对接结果的目录

    filter_rmsd(docked_dir, reference_pdb, output_dir, rmsd_threshold=1.0)
    print("RMSD 筛选完成，保留 RMSD 小于 1.0 的对接结果！")
