import pandas as pd
import numpy as np

def load_and_compare_admet():
    # 读取两个CSV文件
    df1 = pd.read_csv('../data/optimized_ligand/modified_smiles_fg_preds.csv')
    df2 = pd.read_csv('../data/optimized_ligand/original_smiles_fg_preds.csv')
    
    # 定义需要比较的ADMET属性列表
    admet_cols = [
        'AMES', 'BBB_Martins', 'Bioavailability_Ma', 
        'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Substrate_CarbonMangels',
        'CYP2C9_Veith', 'CYP2D6_Substrate_CarbonMangels', 'CYP2D6_Veith',
        'CYP3A4_Substrate_CarbonMangels', 'CYP3A4_Veith', 'Carcinogens_Lagunin',
        'ClinTox', 'DILI', 'HIA_Hou', 'PAMPA_NCATS', 'Pgp_Broccatelli',
        'Skin_Reaction', 'hERG', 'Caco2_Wang', 'Clearance_Hepatocyte_AZ',
        'Clearance_Microsome_AZ', 'Half_Life_Obach', 'HydrationFreeEnergy_FreeSolv',
        'LD50_Zhu', 'Lipophilicity_AstraZeneca', 'PPBR_AZ', 'Solubility_AqSolDB',
        'VDss_Lombardo'
    ]
    
    # 创建结果列表
    results = []
    
    # 对每一对分子进行比较
    for i in range(len(df1)):
        comparison = {}
        comparison['source_smiles'] = df1.iloc[i]['original_smiles']
        comparison['target_smiles'] = df1.iloc[i]['modified_smiles']
        
        # 比较每个ADMET属性
        properties_comparison = []
        for col in admet_cols:
            val1 = df1.iloc[i][col]
            val2 = df2.iloc[i][col]
            
            # 比较值并分类为-1, 0, 1
            if abs(val1 - val2) < 0.1:  # 设置一个小的阈值来判断是否相等
                comp_value = 0
            else:
                comp_value = 1 if val2 > val1 else -1
                
            properties_comparison.append(f"{col}:{comp_value}")
        
        comparison['property_changes'] = ' '.join(properties_comparison)
        results.append(comparison)
    
    # 将结果保存为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存为CSV文件
    output_path = '../data/optimized_ligand/admet_comparison_prompts_fg.csv'
    results_df.to_csv(output_path, index=False)
    print(f"比较结果已保存到: {output_path}")

if __name__ == "__main__":
    load_and_compare_admet()