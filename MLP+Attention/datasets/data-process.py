import os
import pandas as pd

##############################################
#数据特点：
#1.标签在单独一个xlsx表中
#2.提取数据第二列，一个xlsx表为一个样本，将提取的第二列按原始顺序展平整合成一行
##############################################
# 标签文件路径
label_file = r'E:\数据\黄昭景\2025.04.21\12色正交表.xls'
# 数据文件夹根路径
data_root_dir = r'E:\数据\黄昭景\2025.04.21\B组国标浓度\B组国标浓度-未基线校正'

# 读取标签数据
df_label = pd.read_excel(label_file)

# 初始化空列表，用于存储所有数据
all_samples = []
total_files = 0
processed_files = 0
order = 0  # 用于记录数据顺序

# 遍历所有序号文件夹
for index_folder in os.listdir(data_root_dir):
    folder_path = os.path.join(data_root_dir, index_folder)
    if os.path.isdir(folder_path):
        # 获取当前文件夹的序号
        try:
            index = int(index_folder)
        except ValueError:
            print(f"文件夹 {index_folder} 名称不是有效的序号，跳过该文件夹。")
            continue

        # 查找当前序号对应的标签
        label = df_label[df_label['Index'] == index]['Fe3+'].values
        if len(label) > 0:
            label = label[0]
        else:
            print(f"未找到序号 {index} 对应的标签，跳过该文件夹下的文件。")
            continue

        # 查找当前文件夹下的所有 xlsx 文件
        xlsx_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        total_files += len(xlsx_files)
        for xlsx_file in xlsx_files:
            file_path = os.path.join(folder_path, xlsx_file)
            try:
                # 读取 Excel 文件
                df = pd.read_excel(file_path, sheet_name=0, header=None)
                # 提取第 2 列的数据，并且重置索引
                if df.shape[1] > 1:
                    column_data = df.iloc[:, 1].reset_index(drop=True).values.astype(float)
                    # 将数据存入列表，每个元素是一个字典，代表一个样本，并添加顺序标识
                    all_samples.append({
                        'Label': label,
                        'Data': column_data,
                        'Order': order
                    })
                    order += 1
                    processed_files += 1
                else:
                    print(f"文件 {file_path} 中列数不足，无法提取第二列数据")
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

print(f"总共发现 {total_files} 个 xlsx 文件，成功处理 {processed_files} 个文件。")

# 将数据转换为 DataFrame
all_data = pd.DataFrame(all_samples)

# 将 Data 列展开为多列
expanded_data = pd.DataFrame([sample['Data'] for sample in all_samples])

# 合并 Label、展开后的 Data 列和顺序标识列
all_data = pd.concat([all_data[['Label', 'Order']], expanded_data], axis=1)

# 根据顺序标识列对数据进行排序
all_data = all_data.sort_values(by='Order').reset_index(drop=True)

# 删除顺序标识列
all_data = all_data.drop('Order', axis=1)

# 将处理后的结果保存到 xlsx 文件
all_data.to_excel(r"E:\数据\黄昭景\2025.04.21\B组国标浓度\B组国标浓度-未基线校正结果\Fe3+.xlsx", index=False)

print("数据处理完毕，结果已保存到文件")