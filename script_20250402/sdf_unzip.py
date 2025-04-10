import os
import gzip
from rdkit import Chem

# 设置输入和输出目录
input_dir = '../ZINC'
output_dir = '../data/zinc'
allowed_elements = {'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 遍历ZINC目录下的所有.sdf.gz文件
for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith('.sdf.gz'):
            # 构造完整文件路径
            gz_path = os.path.join(root, filename)
            # 提取基础文件名（去除扩展名）
            base_name = os.path.splitext(filename)[0].replace('.sdf', '')
            
            # 使用gzip打开并读取文件
            with gzip.open(gz_path, 'rb') as gz_file:
                # 创建分子供应器
                supplier = Chem.ForwardSDMolSupplier(gz_file)
                molecule_count = 0  # 有效分子计数器
                
                # 遍历每个分子
                for idx, mol in enumerate(supplier):
                    if mol is None:
                        continue  # 跳过无法解析的分子
                    
                    # 检查所有原子是否符合要求
                    valid = True
                    for atom in mol.GetAtoms():
                        symbol = atom.GetSymbol()
                        if symbol not in allowed_elements:
                            valid = False
                            break
                    
                    # 若有效则保存
                    if valid:
                        molecule_count += 1
                        # 生成唯一文件名
                        output_filename = f"{base_name}_mol{molecule_count}_{idx}.sdf"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # 写入分子到文件
                        writer = Chem.SDWriter(output_path)
                        writer.write(mol)
                        writer.close()

            
            print(f"处理完成：{filename}，找到有效分子数：{molecule_count}")

print("全部处理完成！")