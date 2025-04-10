import os
import pandas as pd
from rdkit import Chem



def sdf_filter(dir_path, group_num):
    # 用于保存所有找到的SDF文件和SMILES
    sdf_smiles = []
    sdf_smiles_not_repeat = set()  # 用于存储不重复的SMILES

    # 遍历crossdocked2020文件夹中的所有子文件夹，找到所有SDF文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".sdf"):
                sdf_path = os.path.join(root, file)
                suppl = Chem.SDMolSupplier(sdf_path)

                for mol in suppl:
                    if mol is not None:  # 确保分子读取成功
                        smiles = Chem.MolToSmiles(mol)
                        # 将SMILES添加到集合中，自动去重
                        sdf_smiles.append({"SDF File": file, "SMILES": smiles})


                for mol in suppl:
                    if mol is not None:  # 确保分子读取成功
                        smiles = Chem.MolToSmiles(mol)
                        # 将SMILES添加到集合中，自动去重
                        sdf_smiles_not_repeat.add(smiles)
          
    # 计算总的SDF文件数量
    num_sdf_smiles_not_repeat = len(sdf_smiles_not_repeat)

    # 输出结果
    print(f"总的SDF文件数量: {num_sdf_smiles_not_repeat}")

    sdf_df = pd.DataFrame(sdf_smiles)
    sdf_df.to_csv(f"group_{group_num}.csv", index=False)

    sdf_df_not_repeat = pd.DataFrame(sdf_smiles_not_repeat)
    sdf_df_not_repeat.to_csv(f"group_{group_num}_not_repeat.csv", index=False)


# 设置crossdocked2020文件夹路径和drug_bank.csv文件路径

for group_num in range(2,49):
    dir_path = f"group_{group_num}"

    sdf_filter(dir_path, group_num)