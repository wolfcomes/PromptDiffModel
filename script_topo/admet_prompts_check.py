import pandas as pd
import sys
from datetime import datetime
from io import StringIO

class TeeStream:
    def __init__(self, stdout, file):
        self.stdout = stdout
        self.file = file
        
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def check_admet_comparison():
    # 创建输出文件
    output_file = '../data/optimized_ligand/admet_comparison_info_fg.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        # 记录检查时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"检查时间: {current_time}\n\n")
        
        # 设置输出同时写入文件和控制台
        original_stdout = sys.stdout
        sys.stdout = TeeStream(sys.stdout, f)
        
        try:
            # 读取CSV文件
            file_path = '../data/optimized_ligand/admet_comparison_prompts_fg.csv'
            df = pd.read_csv(file_path)
            
            # 显示基本信息
            print("文件基本信息：")
            buffer = StringIO()
            df.info(buf=buffer)
            print(buffer.getvalue())
            print("\n数据形状：", df.shape)
            
            # 显示列名
            print("\n列名：")
            print(df.columns.tolist())
            
            # 显示前几行数据
            print("\n前2行数据：")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(df.head(2))
            
            # 显示property_changes的示例
            print("\nproperty_changes的第一个样例：")
            changes = df['property_changes'].iloc[0]
            changes_list = changes.split(' ')
            for change in changes_list:
                print(change)
                
            # 添加一些统计信息
            print("\n基本统计信息：")
            print(f"总样本数: {len(df)}")
            
            # 分析property_changes中的变化情况
            first_changes = changes_list[0].split(':')
            print(f"\n每个属性的变化值示例 (以{first_changes[0]}为例)：")
            print("  -1: 属性变差")
            print("   0: 属性基本不变")
            print("   1: 属性变好")
            
        finally:
            # 恢复标准输出
            sys.stdout = original_stdout
            
    print(f"\n信息已保存到: {output_file}")

if __name__ == "__main__":
    check_admet_comparison()