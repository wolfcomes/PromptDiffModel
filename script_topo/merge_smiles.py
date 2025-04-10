import pandas as pd
from rdkit import Chem
import re

# 读取origin_smi文件
origin_smi_file = '../data/crossdocked_groups/group_1.csv'  # 假设文件是CSV格式
origin_smi_df = pd.read_csv(origin_smi_file)

# 读取smiles文件
smiles_file = '../data/generate_data/PPB_data.csv'  # 假设文件是CSV格式
smiles_df = pd.read_csv(smiles_file)



# 合并数据：使用标准化后的SMILES列进行匹配
merged_df = pd.merge(origin_smi_df, smiles_df, left_on='SMILES', right_on='smiles', how='inner')

# 保存合并后的结果
output_file = '../data/generate_data/PPB_data_all.csv'
merged_df.to_csv(output_file, index=False)

print(f'Merged data saved to {output_file}')