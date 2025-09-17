import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import shap
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')



# ==============================
# 1. 配置参数
# ==============================
class Config:
    def __init__(self):
        # # 数据路径，划分好训练集和测试集
        self.train_path = r"E:\数据\刘+黄\Ag总.xlsx"
        self.test_path = r"E:\数据\刘+黄\刘+黄校验集\真实样本-Ag总.xlsx"

        # # 数据路径，这里只需要原始数据路径，因为要自动划分训练集和测试集
        # self.data_path = r"E:\数据\黄昭景\2025.04.21\A+B格式校正+紫外\未基线校正\Pb总.xlsx"

        # 数据列配置（根据实际Excel列结构修改）
        self.label_col = 0  # 标签列索引
        self.fl_range = (0, 284)  # 拉曼数据列范围（左闭右开）
        self.uv_range = (284, 684)  # 紫外数据列范围（左闭右开）

        # 计算输入特征维度
        self.input_size_fl = self.fl_range[1] - self.fl_range[0]
        self.input_size_uv = self.uv_range[1] - self.uv_range[0]

        # 模型参数
        self.output_dir = r'E:\数据\刘+黄结果\Cd'
        self.model_save_path = os.path.join(self.output_dir, 'model.pth')
        self.batch_size = 32
        self.num_epochs = 250
        self.learning_rate = 0.001
        self.early_stop_patience = 40
        self.hidden_size = 64
        self.num_classes = 2  # 根据实际类别数修改
        self.use_cuda = torch.cuda.is_available()

        #测试集比例，这里设置为0.2，即20%的数据作为测试集
        self.test_size = 0.2


config = Config()

# ==============================
# 2. 自定义数据集类
# ==============================
class SpectralDataset(Dataset):
    def __init__(self, features, labels, fl_scaler=None, uv_scaler=None):
        """
        :param features: 原始特征数据（未分割标准化）
        :param labels: 标签数据
        :param fl_scaler: 荧光数据标准化器（训练集传入None自动创建）
        :param uv_scaler: 紫外数据标准化器（训练集传入None自动创建）
        """
        # 分割数据
        self.fl_data = features[:, config.fl_range[0]:config.fl_range[1]]
        self.uv_data = features[:, config.uv_range[0]:config.uv_range[1]]

        # 标准化处理
        if fl_scaler is None:  # 训练模式
            self.fl_scaler = StandardScaler()
            self.uv_scaler = StandardScaler()
            self.fl_data = self.fl_scaler.fit_transform(self.fl_data)
            self.uv_data = self.uv_scaler.fit_transform(self.uv_data)
        else:  # 测试模式
            self.fl_scaler = fl_scaler
            self.uv_scaler = uv_scaler
            self.fl_data = fl_scaler.transform(self.fl_data)
            self.uv_data = uv_scaler.transform(self.uv_data)

        # 合并标准化后的数据
        self.features = np.concatenate([self.fl_data, self.uv_data], axis=1)
        # # 可以添加打印语句检查合并后数据的形状
        # print(f"self.features shape: {self.features.shape}")
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        return features, label.squeeze()

# ==============================
# 3. 数据加载函数
# ==============================
def load_data():
    #########自己分训练集和测试集##########
    # # 加载原始数据
    train_df = pd.read_excel(config.train_path, engine='openpyxl')
    test_df = pd.read_excel(config.test_path, engine='openpyxl')
    # #提取特征和标签
    X_train = train_df.iloc[:, 1:].values  # 跳过标签列# 第1列之后的所有列作为特征
    y_train = train_df.iloc[:, config.label_col].values
    X_test = test_df.iloc[:, 1:].values
    y_test = test_df.iloc[:, config.label_col].values

    # ######################程序分训练集和测试集########
    # # 加载原始数据
    # df = pd.read_excel(config.data_path, engine='openpyxl')
    # # 提取特征和标签
    # X = df.iloc[:, 1:].values  # 跳过标签列# 第1列之后的所有列作为特征
    # y = df.iloc[:, config.label_col].values
    # # 使用train_test_split自动划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=42)
    ###########################################################
    #######检查划分是否正确
    # # 打印训练集和测试集的特征和标签形状，
    # print("训练集特征形状:", X_train.shape)
    # print("训练集标签形状:", y_train.shape)
    # print("测试集特征形状:", X_test.shape)
    # print("测试集标签形状:", y_test.shape)
    #
    # # 检查荧光数据和紫外数据的提取是否正确，打印部分数据
    # print("训练集前5个样本的荧光数据:")
    # train_fl_data = X_train[:, config.fl_range[0]:config.fl_range[1]]
    # print(train_fl_data[:5])
    #
    # print("训练集前5个样本的紫外数据:")
    # train_uv_data = X_train[:, config.uv_range[0]:config.uv_range[1]]
    # print(train_uv_data[:5])
    # print("训练集紫外数据形状:", train_uv_data.shape)  # 添加这一行打印形状
    #
    # print("测试集前5个样本的荧光数据:")
    # test_fl_data = X_test[:, config.fl_range[0]:config.fl_range[1]]
    # print(test_fl_data[:5])
    #
    # print("测试集前5个样本的紫外数据:")
    # test_uv_data = X_test[:, config.uv_range[0]:config.uv_range[1]]
    # print(test_uv_data[:5])
    # print("测试集紫外数据形状:", test_uv_data.shape)  # 添加这一行打印形状

    # 创建数据集（只在训练集上拟合scaler）
    train_dataset = SpectralDataset(X_train, y_train)
    test_dataset = SpectralDataset(X_test, y_test,
                                  train_dataset.fl_scaler,
                                  train_dataset.uv_scaler)
    return train_dataset, test_dataset

# ==============================
# 4. 模型定义 (MLP + Attention)
# ==============================

class MLPAttention(torch.nn.Module):
    def __init__(self, input_size_fl, input_size_uv, num_classes):
        super().__init__()
        # 荧光分支
        self.fl_fc1 = torch.nn.Linear(input_size_fl, config.hidden_size)
        self.fl_fc2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        # 紫外分支
        # print(f"input_size_uv: {input_size_uv}")  # 添加这一行打印 input_size_uv 的值
        self.uv_fc1 = torch.nn.Linear(input_size_uv, config.hidden_size)
        self.uv_fc2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        # 注意力机制
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size * 2, 1),
            torch.nn.Sigmoid()
        )

        # 分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size * 2, config.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(config.hidden_size, num_classes)
        )

    def forward(self, x):
        # 分割输入数据
        fl_data = x[:, :config.input_size_fl]
        uv_data = x[:, config.input_size_fl:]
        # print(f"fl_data shape: {fl_data.shape}")
        # print(f"uv_data shape before fc1: {uv_data.shape}")  # 添加这一行打印 uv_data 的初始维度
        # 荧光处理
        fl = torch.relu(self.fl_fc1(fl_data))
        fl = torch.relu(self.fl_fc2(fl))

        # 紫外处理
        uv = torch.relu(self.uv_fc1(uv_data))
        uv = torch.relu(self.uv_fc2(uv))
        # print(f"uv shape after fc1: {uv.shape}")  # 添加这一行打印 uv 经过 fc1 后的维度

        # # 打印中间结果的形状，用于调试
        # print(f"fl shape: {fl.shape}")
        # print(f"uv shape: {uv.shape}")

        # 注意力融合
        combined = torch.cat([fl, uv], dim=1)
        # print(f"combined shape: {combined.shape}")
        attention_weights = self.attention(combined)
        weighted_combined = attention_weights * fl + (1 - attention_weights) * uv

        # 分类
        output = self.classifier(combined)
        return output


# ==============================
# 5. 训练器类
# ==============================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if config.use_cuda else "cpu")

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 加载数据集
        self.train_dataset, self.test_dataset = load_data()

        # 初始化模型
        self.model = MLPAttention(config.input_size_fl, config.input_size_uv, config.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        best_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        train_accuracies = []

        # 打开文件用于写入训练信息
        log_file_path = os.path.join(self.config.output_dir, 'training_log.txt')
        with open(log_file_path, 'w') as log_file:
            for epoch in range(config.num_epochs):
                self.model.train()
                epoch_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(self.device), targets.to(self.device)

                    # 前向传播
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 统计指标
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                # 计算epoch指标
                avg_loss = epoch_loss / len(train_loader)
                accuracy = 100.0 * correct / total
                train_losses.append(avg_loss)
                train_accuracies.append(accuracy)

                # 学习率调度
                self.scheduler.step(avg_loss)

                # 打印信息并写入文件
                epoch_info = (
                    f'Epoch [{epoch + 1}/{config.num_epochs}]\n'
                    f'Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n'
                    f'Current LR: {self.optimizer.param_groups[0]["lr"]:.6f}\n'
                )
                print(epoch_info)
                log_file.write(epoch_info)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    early_stop_counter = 0
                    torch.save(self.model.state_dict(), config.model_save_path)
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= config.early_stop_patience:
                        print(f'Early stopping at epoch {epoch + 1}')
                        log_file.write(f'Early stopping at epoch {epoch + 1}\n')
                        break

        # 绘制训练曲线
        self.plot_metrics(train_accuracies, train_losses)

        return train_accuracies, train_losses

    def test(self):
        self.model.load_state_dict(torch.load(config.model_save_path))
        self.model.eval()
        test_loader = DataLoader(self.test_dataset, batch_size=config.batch_size, shuffle=False)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_preds)
        print(f'Test Accuracy: {accuracy:.4f}')

        # 保存预测结果
        results_df = pd.DataFrame({
            'True': all_targets,
            'Predicted': all_preds
        })
        results_df.to_csv(os.path.join(config.output_dir, 'predictions.csv'), index=False)

        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_targets, all_preds)

        return accuracy, all_preds, all_targets

    def plot_metrics(self, accuracies, losses):
        # 绘制训练精度
        plt.plot(accuracies, label='Train Accuracy', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy', color='blue')
        plt.tick_params(axis='y', labelcolor='blue')
        plt.legend(loc='upper left')

        # 创建第二个y轴并绘制训练损失
        plt.twinx()
        plt.plot(losses, label='Train Loss', color='red')
        plt.ylabel('Loss', color='red')
        plt.tick_params(axis='y', labelcolor='red')
        plt.legend(loc='upper right')
        # 设置x轴刻度为整数，并以25的间距显示，并且保证出图的x轴刻度会多一个间距出图使图更好看
        plt.xticks(np.arange(0, len(accuracies) + 4, 10))

        plt.title('Training Accuracy and Loss')
        plt.tight_layout()
        # 保存图片到指定文件路径
        #plt.savefig(r'E:\数据\刘+黄结果\Ba\Training Accuracy and Loss.png')
        image_path = os.path.join(self.config.output_dir, 'Training Accuracy and Loss.png')
        plt.savefig(image_path)
        #plt.show()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, ax=ax, cmap=sns.color_palette("YlOrRd", as_cmap=True), fmt='g')
        # 修改左侧y轴刻度方向
        ax.invert_yaxis()
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        # 保存图片到指定文件路径
        #plt.savefig(r'E:\数据\刘+黄结果\AL\Confusion Matrix.png')
        image_path = os.path.join(self.config.output_dir, 'Confusion Matrix.png')
        plt.savefig(image_path)
        #plt.show()


#横坐标值更大，意味着这些特征对模型输出的影响更大，通过蓝色和红色部分的长度对比，可以看出该特征对不同类别的影响差异
#即SHAP可解释性分析解释了在模型中哪些特征更重要，以及这些特征对不同类别预测的相对贡献情况
    def explain_model(self, sample_idx=0):
        # SHAP可解释性分析
        background = torch.stack([self.train_dataset[i][0] for i in range(10)]).to(self.device)
        test_sample = torch.stack([self.test_dataset[sample_idx][0]]).to(self.device)

        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(test_sample,check_additivity=False )

        # 可视化SHAP值
        plt.figure()
        shap.summary_plot(shap_values, test_sample.cpu().numpy(),
                          feature_names=[f"FL_{i}" for i in range(config.input_size_fl)] +
                                        [f"UV_{i}" for i in range(config.input_size_uv)],
                          plot_type='bar')
        plt.pause(0.1)  # 添加这一行
        image_path = os.path.join(self.config.output_dir, 'shap_summary.png')
        plt.savefig(image_path)
        #plt.savefig(r'datasets\shap_summary.png')
        plt.close()





# ==============================
# 5. 执行训练和测试
# ==============================
if __name__ == "__main__":
    trainer = Trainer(config)
    train_acc, train_loss = trainer.train()
    test_acc, preds, trues = trainer.test()
    trainer.explain_model()

