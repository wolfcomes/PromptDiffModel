import pandas as pd
import os
import glob

# 存储所有读取的group文件
all_groups_df = pd.DataFrame()

# ADMET数据
admet_file = "/data/home/zhangzhiyong/lead_optimization/PromptDiffModel/data/optimized_ligand/admet_comparison_prompts_fg.csv"
admet_df = pd.read_csv(admet_file)
print(f"ADMET CSV包含 {len(admet_df)} 行数据")

# 遍历group_1到group_48的所有文件
base_path = "/data/home/zhangzhiyong/lead_optimization/PromptDiffModel/data/crossdocked_groups"
total_group_records = 0

for group_num in range(1, 49):
    group_file = os.path.join(base_path, f"group_{group_num}.csv")
    
    # 检查文件是否存在
    if os.path.exists(group_file):
        # 读取当前group文件
        group_df = pd.read_csv(group_file)
        
        # 添加group列
        group_df['group'] = group_num
        
        # 重命名列以便合并
        group_df = group_df.rename(columns={"SMILES": "source_smiles"})
        
        # 添加到总DataFrame
        all_groups_df = pd.concat([all_groups_df, group_df], ignore_index=True)
        
        total_group_records += len(group_df)
        print(f"读取 group_{group_num}.csv，包含 {len(group_df)} 行")

print(f"总共读取了 {total_group_records} 条group记录")

# 进行合并 - 以source_smiles为键
merged_df = pd.merge(
    admet_df, 
    all_groups_df, 
    on="source_smiles", 
    how="inner"  # 只保留匹配的行
)

# 调整列顺序
merged_df = merged_df[['group', 'SDF File', 'source_smiles', 'target_smiles', 'property_changes']]

# 保存结果
output_file = "/data/home/zhangzhiyong/lead_optimization/PromptDiffModel/data/optimized_ligand/merged_all_groups_admet.csv"
merged_df.to_csv(output_file, index=False)

# 打印匹配情况统计
print(f"\n所有Group CSV总共包含 {total_group_records} 行数据")
print(f"ADMET CSV包含 {len(admet_df)} 行数据")
print(f"合并后的文件包含 {len(merged_df)} 行匹配数据")
print(f"结果已保存至: {output_file}")

# 分析每个group的匹配情况
group_counts = merged_df['group'].value_counts().sort_index()
print("\n各group匹配记录数:")
for group, count in group_counts.items():
    print(f"Group {group}: {count}条记录")