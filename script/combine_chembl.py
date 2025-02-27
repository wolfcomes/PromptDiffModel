import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter

def load_and_combine_batches(processed_dir, split):
    """
    加载并合并指定 split 的所有批次数据。
    """
    # 获取所有批次的文件
    batch_files = sorted(processed_dir.glob(f"{split}_batch_*.npz"))
    if not batch_files:
        raise FileNotFoundError(f"No batch files found for split '{split}' in {processed_dir}")

    # 初始化空列表以存储合并后的数据
    combined_opt_lig_coords = []
    combined_opt_lig_one_hot = []
    combined_opt_lig_mask = []
    combined_pocket_coords = []
    combined_pocket_one_hot = []
    combined_pocket_mask = []
    combined_mol_ids = []

    # 遍历所有批次文件并加载数据
    for batch_file in batch_files:
        data = np.load(batch_file, allow_pickle=True)
        combined_opt_lig_coords.append(data['opt_lig_coords'])
        combined_opt_lig_one_hot.append(data['opt_lig_one_hot'])
        combined_opt_lig_mask.append(data['opt_lig_mask'])
        combined_pocket_coords.append(data['pocket_coords'])
        combined_pocket_one_hot.append(data['pocket_one_hot'])
        combined_pocket_mask.append(data['pocket_mask'])
        combined_mol_ids.extend(data['names'])

    # 合并数据
    combined_opt_lig_coords = np.concatenate(combined_opt_lig_coords, axis=0)
    combined_opt_lig_one_hot = np.concatenate(combined_opt_lig_one_hot, axis=0)
    combined_opt_lig_mask = np.concatenate(combined_opt_lig_mask, axis=0)
    combined_pocket_coords = np.concatenate(combined_pocket_coords, axis=0)
    combined_pocket_one_hot = np.concatenate(combined_pocket_one_hot, axis=0)
    combined_pocket_mask = np.concatenate(combined_pocket_mask, axis=0)

    combined_output_path = processed_dir / f"{split}_combined.npz"
    np.savez(
        combined_output_path,
        opt_lig_coords=combined_opt_lig_coords,
        opt_lig_one_hot=combined_opt_lig_one_hot,
        opt_lig_mask=combined_opt_lig_mask,
        pocket_coords=combined_pocket_coords,
        pocket_one_hot=combined_pocket_one_hot,
        pocket_mask=combined_pocket_mask,
        names=combined_mol_ids
    )
    print(f"Combined data saved to {combined_output_path}")

    # 返回合并后的数据
    return {
        'opt_lig_coords': combined_opt_lig_coords,
        'opt_lig_one_hot': combined_opt_lig_one_hot,
        'opt_lig_mask': combined_opt_lig_mask,
        'pocket_coords': combined_pocket_coords,
        'pocket_one_hot': combined_pocket_one_hot,
        'pocket_mask': combined_pocket_mask,
        'mol_ids': combined_mol_ids
    }

def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)
    print(f"idx_lig: {len(idx_lig)}")
    print(f"idx_pocket: {len(idx_pocket)}")
    assert np.all(idx_lig == idx_pocket)

    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram

def generate_size_distribution(processed_dir, split):
    """
    生成 size_distribution.npy 文件。
    """
    # 加载并合并数据
    combined_data = load_and_combine_batches(processed_dir, split)

    # 调用 get_n_nodes 函数
    n_nodes = get_n_nodes(combined_data['opt_lig_mask'], combined_data['pocket_mask'], smooth_sigma=1.0)

    # 保存结果
    output_path = Path(processed_dir, 'size_distribution.npy')
    np.save(output_path, n_nodes)
    print(f"Size distribution saved to {output_path}")


if __name__ == '__main__':
    # 设置 processed_dir 和 split
    processed_dir = Path("../data/chembl_npz")  # 替换为实际的路径
    split = "train"  # 可以是 "train", "val", 或 "test"

    # 生成 size_distribution.npy
    generate_size_distribution(processed_dir, split)