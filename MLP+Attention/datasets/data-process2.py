import os
import pandas as pd

###################################################
#将txt文件转为xlsx文件
###################################################

# 指定包含 txt 文件的文件夹路径
folder_path = r'E:\数据\黄昭景\2025.04.21\B组国标浓度\B组国标浓度-未基线校正\24'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # 构建 txt 文件的完整路径
        txt_file_path = os.path.join(folder_path, filename)
        # 构建对应的 xlsx 文件的完整路径，替换扩展名
        xlsx_file_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.xlsx')

        try:
            # 读取 txt 文件，使用 '\s+' 匹配一个或多个空白字符作为分隔符，并指定编码
            df = pd.read_csv(txt_file_path, sep='\s+', header=None, encoding='utf-8')
            # 将 DataFrame 写入 xlsx 文件
            df.to_excel(xlsx_file_path, index=False, header=False)
            print(f"已成功将 {txt_file_path} 转换为 {xlsx_file_path}")
        except Exception as e:
            print(f"转换 {txt_file_path} 时出错: {e}")
