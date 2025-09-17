import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve,
                             auc, precision_recall_curve, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef,
                             cohen_kappa_score, log_loss, classification_report)
import shap
from sklearn.model_selection import train_test_split
import matplotlib
from scipy.signal import savgol_filter  # 新增SG平滑依赖

matplotlib.use('Agg')
# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

###############################################################
#Test5是以Test2为基础进行的修改并且shap可解释性分析图也进行了修改
#Test5对比main,Test2对拉曼光谱做了针对性优化,如果追求公平对比，应使用main,Test2；如果希望突出拉曼光谱的贡献，应使用Test5。
###############################################################

# ==============================
# 1. 配置参数
# ==============================
class Config:
    def __init__(self):
        # 数据路径
        self.data_path = r"E:\数据\黄昭景\2025.04.21\A+B格式校正+紫外\未基线校正\Zn总.xlsx"

        # 数据列配置
        self.label_col = 0  # 标签列索引
        self.fl_range = (0, 2400)  # 拉曼数据列范围
        self.uv_range = (2400, 2801)  # 紫外数据列范围

        # 计算输入特征维度
        self.input_size_fl = self.fl_range[1] - self.fl_range[0]
        self.input_size_uv = self.uv_range[1] - self.uv_range[0]

        # 模型参数
        self.output_dir = r'E:\数据\黄昭景\2025.04.21\多模态高浓度特征偏拉曼数据\结果\Zn'
        self.model_save_path = os.path.join(self.output_dir, 'model.pth')
        self.batch_size = 32
        self.num_epochs = 200
        self.learning_rate = 0.001
        self.early_stop_patience = 50
        self.hidden_size = 64
        self.num_classes = 2
        self.use_cuda = torch.cuda.is_available()
        self.test_size = 0.2

        # 预处理参数（两种模式共用）
        self.smoothing_window = 7  # 公平模式下的统一平滑窗口
        self.smoothing_polyorder = 3  # 公平模式下的统一多项式阶数
        self.raman_smoothing_window = 9  # 偏向模式下拉曼平滑窗口（原为7，可增大为9使平滑更剧烈）
        self.uv_smoothing_window = 7  # 偏向模式下紫外平滑窗口（原为5，增大到7与拉曼一致）


config = Config()


# ==============================
# 2. 自定义数据集类
# ==============================
class SpectralDataset(Dataset):
    def __init__(self, features, labels, fl_scaler=None, uv_scaler=None):
        # 分割数据
        self.fl_data = features[:, config.fl_range[0]:config.fl_range[1]]  # 拉曼数据
        self.uv_data = features[:, config.uv_range[0]:config.uv_range[1]]  # 紫外数据

        # ==============================================
        # 公平预处理（对称处理）：拉曼和紫外采用完全相同的预处理
        # ==============================================
        # # 平滑去噪（统一参数）
        # self.fl_data = self.smooth_spectra(self.fl_data, window=config.smoothing_window)
        # self.uv_data = self.smooth_spectra(self.uv_data, window=config.smoothing_window)
        #
        # # 标准化（统一方法：均用StandardScaler）
        # if fl_scaler is None:  # 训练模式
        #     self.fl_scaler = StandardScaler()
        #     self.uv_scaler = StandardScaler()
        #     self.fl_data = self.fl_scaler.fit_transform(self.fl_data)
        #     self.uv_data = self.uv_scaler.fit_transform(self.uv_data)
        # else:  # 测试模式
        #     self.fl_data = fl_scaler.transform(self.fl_data)
        #     self.uv_data = uv_scaler.transform(self.uv_data)

        # ==============================================
        # 偏向拉曼预处理：针对拉曼特性优化，与紫外处理不同
        # ==============================================
        # 拉曼数据预处理
        self.fl_data = self.baseline_correction(self.fl_data)  # 基线校正（仅拉曼）
        self.fl_data = self.smooth_spectra(self.fl_data, window=config.raman_smoothing_window)  # 拉曼平滑
        # 峰高归一化（改为可选，降低偏向强度）
        #self.fl_data = self.normalize_peak(self.fl_data)  # 峰高归一化（仅拉曼）

        # 紫外数据预处理
        #self.uv_data = self.correct_scattering(self.uv_data)  # 散射校正
        self.uv_data = self.baseline_correction(self.uv_data)  # 新增：对紫外也做基线校正
        self.uv_data = self.smooth_spectra(self.uv_data, window=config.uv_smoothing_window)  # 紫外平滑（参数不同）
        # 新增：对紫外也做峰高归一化（进一步平衡）
        self.uv_data = self.normalize_peak(self.uv_data)

        # 标准化处理（差异化标准化）,原为拉曼StandardScaler+紫外MinMaxScaler
        if fl_scaler is None:  # 训练模式
            self.fl_scaler = StandardScaler()  # 拉曼用StandardScaler
            #self.uv_scaler = MinMaxScaler()    # 紫外用MinMaxScaler
            self.uv_scaler = StandardScaler()
            self.fl_data = self.fl_scaler.fit_transform(self.fl_data)
            self.uv_data = self.uv_scaler.fit_transform(self.uv_data)
        else:  # 测试模式
            self.fl_scaler = fl_scaler
            self.uv_scaler = uv_scaler
            self.fl_data = fl_scaler.transform(self.fl_data)
            self.uv_data = uv_scaler.transform(self.uv_data)
        # # ==============================================

        # 合并标准化后的数据
        self.features = np.concatenate([self.fl_data, self.uv_data], axis=1)
        self.labels = labels

    # 以下为预处理工具函数（偏向模式使用）
    def baseline_correction(self, spectra):
        """多项式基线校正（去除荧光背景）"""
        baseline = np.zeros_like(spectra)
        for i in range(len(spectra)):
            x = np.arange(len(spectra[i]))
            p = np.polyfit(x, spectra[i], 3)  # 3次多项式拟合基线
            baseline[i] = np.polyval(p, x)
        return spectra - baseline

    def smooth_spectra(self, spectra, window, polyorder=3):
        """SG平滑去噪（两种模式共用，参数可不同）"""
        if window % 2 == 0:  # 确保窗口为奇数
            window += 1
        return savgol_filter(spectra, window_length=window, polyorder=polyorder, axis=1)

    def normalize_peak(self, spectra):
        """拉曼峰高归一化（保留峰强度相对差异）（拉曼专用）"""
        max_values = np.max(spectra, axis=1, keepdims=True)
        max_values[max_values == 0] = 1  # 避免除零错误
        return spectra / max_values

    def correct_scattering(self, spectra):
        """紫外数据散射校正（SNV算法）（紫外专用）"""
        mean = np.mean(spectra, axis=1, keepdims=True)
        std = np.std(spectra, axis=1, keepdims=True)
        std[std == 0] = 1  # 避免除零错误
        return (spectra - mean) / std

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
    # 加载原始数据
    df = pd.read_excel(config.data_path, engine='openpyxl')

    # 提取特征和标签
    X = df.iloc[:, 1:].values
    y = df.iloc[:, config.label_col].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=42, stratify=y
    )

    # 创建数据集
    train_dataset = SpectralDataset(X_train, y_train)
    test_dataset = SpectralDataset(X_test, y_test,
                                   train_dataset.fl_scaler,
                                   train_dataset.uv_scaler)
    return train_dataset, test_dataset


# ==============================
# 4. 模型定义
# ==============================
class MLPAttention(torch.nn.Module):
    def __init__(self, input_size_fl, input_size_uv, num_classes):
        super().__init__()
        # 荧光分支
        self.fl_fc1 = torch.nn.Linear(input_size_fl, config.hidden_size)
        self.fl_fc2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        # 紫外分支
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

        # 荧光处理
        fl = torch.relu(self.fl_fc1(fl_data))
        fl = torch.relu(self.fl_fc2(fl))

        # 紫外处理
        uv = torch.relu(self.uv_fc1(uv_data))
        uv = torch.relu(self.uv_fc2(uv))

        # 注意力融合
        combined = torch.cat([fl, uv], dim=1)
        attention_weights = self.attention(combined)
        weighted_combined = attention_weights * fl + (1 - attention_weights) * uv

        # 分类
        output = self.classifier(combined)
        return output


# ==============================
# 5. 训练器类（核心修改SHAP分析部分）
# ==============================
class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if config.use_cuda else "cpu")

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 加载数据集
        self.train_dataset, self.test_dataset = load_data()

        # 检查数据集类别分布
        self.check_class_distribution()

        # 初始化模型
        self.model = MLPAttention(config.input_size_fl, config.input_size_uv, config.num_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)

    def check_class_distribution(self):
        """检查训练集和测试集的类别分布"""
        train_labels = self.train_dataset.labels
        test_labels = self.test_dataset.labels

        print("\n类别分布检查:")
        print("=" * 40)
        print(f"训练集样本数: {len(train_labels)}")
        print(f"类别0数量: {sum(train_labels == 0)}")
        print(f"类别1数量: {sum(train_labels == 1)}")

        print(f"\n测试集样本数: {len(test_labels)}")
        print(f"类别0数量: {sum(test_labels == 0)}")
        print(f"类别1数量: {sum(test_labels == 1)}")
        print("=" * 40 + "\n")

        # 如果测试集只有一个类别，发出警告
        if len(np.unique(test_labels)) < 2:
            print("警告: 测试集只包含一个类别，某些评估指标可能无法计算!")

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=config.batch_size, shuffle=True)
        best_loss = float('inf')
        early_stop_counter = 0
        train_losses = []
        train_accuracies = []

        # 训练日志文件
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
        all_probs = []  # 保存预测概率

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)  # 获取概率
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算各种指标
        metrics = self.calculate_metrics(all_targets, all_preds, all_probs)

        # 保存预测结果和概率
        results_df = pd.DataFrame({
            'True': all_targets,
            'Predicted': all_preds,
            'Prob_Class0': [p[0] for p in all_probs],
            'Prob_Class1': [p[1] for p in all_probs]
        })
        results_df.to_csv(os.path.join(config.output_dir, 'predictions.csv'), index=False)

        # 绘制评估图表
        self.plot_confusion_matrix(all_targets, all_preds)

        # 只有在测试集包含两个类别时才绘制ROC和PR曲线
        if len(np.unique(all_targets)) >= 2:
            self.plot_roc_curve(all_targets, all_probs)
            self.plot_pr_curve(all_targets, all_probs)
        else:
            print("测试集只包含一个类别，跳过ROC和PR曲线的绘制")

        return metrics, all_preds, all_targets, all_probs

    def calculate_metrics(self, y_true, y_pred, y_probs):
        """计算多种评估指标"""
        metrics = {}

        # 基本分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # 概率评估指标
        if len(np.unique(y_true)) >= 2:
            y_probs_class1 = [p[1] for p in y_probs]  # 正类概率
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs_class1)
            metrics['pr_auc'] = average_precision_score(y_true, y_probs_class1)
        else:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')

        try:
            metrics['log_loss'] = log_loss(y_true, y_probs)
        except:
            metrics['log_loss'] = float('nan')

        # 一致性指标
        try:
            metrics['matthews_corr'] = matthews_corrcoef(y_true, y_pred)
        except:
            metrics['matthews_corr'] = float('nan')

        try:
            metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        except:
            metrics['cohen_kappa'] = float('nan')

        # 分类报告
        metrics['classification_report'] = classification_report(y_true, y_pred)

        # 保存指标到文件
        with open(os.path.join(config.output_dir, 'evaluation_metrics.txt'), 'w') as f:
            f.write("模型评估指标:\n")
            f.write("=" * 50 + "\n")
            f.write(f"准确率 (Accuracy): {metrics['accuracy']:.4f}\n")
            f.write(f"平衡准确率 (Balanced Accuracy): {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"精确率 (Precision): {metrics['precision']:.4f}\n")
            f.write(f"召回率 (Recall): {metrics['recall']:.4f}\n")
            f.write(f"F1分数: {metrics['f1']:.4f}\n")

            if not np.isnan(metrics['roc_auc']):
                f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"PR AUC: {metrics['pr_auc']:.4f}\n")
            else:
                f.write("ROC AUC: 无法计算 (测试集只包含一个类别)\n")
                f.write("PR AUC: 无法计算 (测试集只包含一个类别)\n")

            if not np.isnan(metrics['log_loss']):
                f.write(f"对数损失 (Log Loss): {metrics['log_loss']:.4f}\n")
            else:
                f.write("对数损失 (Log Loss): 无法计算\n")

            if not np.isnan(metrics['matthews_corr']):
                f.write(f"马修斯相关系数 (MCC): {metrics['matthews_corr']:.4f}\n")
            else:
                f.write("马修斯相关系数 (MCC): 无法计算\n")

            if not np.isnan(metrics['cohen_kappa']):
                f.write(f"科恩卡帕系数 (Cohen's Kappa): {metrics['cohen_kappa']:.4f}\n")
            else:
                f.write("科恩卡帕系数 (Cohen's Kappa): 无法计算\n")

            f.write("\n分类报告:\n")
            f.write(metrics['classification_report'])

        # 打印主要指标
        print("\n" + "=" * 50)
        print("模型评估指标:")
        print("=" * 50)
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"平衡准确率 (Balanced Accuracy): {metrics['balanced_accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")

        if not np.isnan(metrics['roc_auc']):
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
            print(f"PR AUC: {metrics['pr_auc']:.4f}")
        else:
            print("ROC AUC: 无法计算 (测试集只包含一个类别)")
            print("PR AUC: 无法计算 (测试集只包含一个类别)")

        if not np.isnan(metrics['log_loss']):
            print(f"对数损失 (Log Loss): {metrics['log_loss']:.4f}")
        else:
            print("对数损失 (Log Loss): 无法计算")

        if not np.isnan(metrics['matthews_corr']):
            print(f"马修斯相关系数 (MCC): {metrics['matthews_corr']:.4f}")
        else:
            print("马修斯相关系数 (MCC): 无法计算")

        if not np.isnan(metrics['cohen_kappa']):
            print(f"科恩卡帕系数 (Cohen's Kappa): {metrics['cohen_kappa']:.4f}")
        else:
            print("科恩卡帕系数 (Cohen's Kappa): 无法计算")

        print("\n分类报告:\n", metrics['classification_report'])

        return metrics

    def plot_metrics(self, accuracies, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, label='Train Accuracy', color='blue', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', color='blue', fontsize=12)
        plt.tick_params(axis='y', labelcolor='blue')
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        ax2 = plt.twinx()
        ax2.plot(losses, label='Train Loss', color='red', linewidth=2)
        ax2.set_ylabel('Loss', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper right', fontsize=10)
        plt.xticks(np.arange(0, len(accuracies) + 1, 10))

        plt.title('训练精度和损失曲线', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'Training Accuracy and Loss.png'), dpi=300)
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    cbar=False, annot_kws={"size": 14})
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title('混淆矩阵', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'Confusion Matrix.png'), dpi=300)
        plt.close()

    def plot_roc_curve(self, y_true, y_probs):
        y_probs_class1 = [p[1] for p in y_probs]  # 正类概率
        fpr, tpr, _ = roc_curve(y_true, y_probs_class1)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率', fontsize=12)
        plt.ylabel('真正例率', fontsize=12)
        plt.title('ROC曲线', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'ROC Curve.png'), dpi=300)
        plt.close()

    def plot_pr_curve(self, y_true, y_probs):
        y_probs_class1 = [p[1] for p in y_probs]  # 正类概率
        precision, recall, _ = precision_recall_curve(y_true, y_probs_class1)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AUC = {pr_auc:.4f})')
        plt.xlabel('召回率', fontsize=12)
        plt.ylabel('精确率', fontsize=12)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('精确率-召回率曲线', fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'PR Curve.png'), dpi=300)
        plt.close()

##################################
# SHAP_Summary_Dot.png（SHAP 值分布点图）
# 含义：
# 图表以散点形式展示每个特征在多个样本上的 SHAP 值分布，是对特征重要性更细致的解读：
# y 轴：特征（按重要性排序，与条形图一致）。
# x 轴：SHAP 值（特征对模型预测的影响大小和方向：正值表示该特征推动预测向 “类别 1” 倾斜，负值表示推动向 “类别 0” 倾斜）。
# 点的颜色：特征自身的数值（颜色越深表示特征值越大），可观察特征值与 SHAP 值的关系（如 “特征值越大，对预测的正向影响越强”）。
# 作用：不仅能看出特征的重要性，还能分析特征值如何影响模型决策（如某个拉曼特征在高值时是否总是推动模型预测为正类）。

# SHAP_Decision_Plot.png（SHAP 决策解释图）
# 针对样本范围：测试集的第一个样本（代码中shap_values[1][0] 取第一个测试样本的类别1 SHAP 值）。
# 含义：图表展示单个样本的决策路径，即每个特征如何一步步影响模型的最终预测：
# 基线值（Base Value）：图表最左侧的横线，代表模型在 “背景样本”（训练集前 200个样本）上的平均预测概率（针对类别1）。
# 特征影响：每个特征用箭头表示其对预测的影响（红色箭头为正向推动，蓝色为负向推动），箭头长度表示SHAP 值大小（影响程度）。
# 最终预测：图表最右侧的点，代表模型对该样本的最终预测概率（类别 1）。
# 作用：直观解释 “模型为何将这个样本预测为某一类”，例如 “该样本的拉曼特征 xxx 值较高，显著推动预测为正类”。
##################################

    def explain_model(self):
        """执行SHAP可解释性分析（修改为：所有测试样本+前5个样本决策路径）"""
        try:
            # 创建特征名称（区分拉曼和紫外）
            fl_feature_names = [f"Raman_{i}" for i in range(config.input_size_fl)]
            uv_feature_names = [f"UV_{i}" for i in range(config.input_size_uv)]
            all_feature_names = fl_feature_names + uv_feature_names

            # 1. 准备数据：使用所有测试样本（而非仅前5个）
            # 背景数据（训练集前200个样本）
            background_indices = range(min(200, len(self.train_dataset)))
            background_data = torch.stack(
                [self.train_dataset[i][0] for i in background_indices]
            ).to(self.device)

            # 所有测试样本（关键修改点1：从测试集前5个改为所有测试样本）
            test_indices = range(len(self.test_dataset))  # 所有测试样本
            test_samples = torch.stack(
                [self.test_dataset[i][0] for i in test_indices]
            ).to(self.device)

            # 2. 创建解释器并计算SHAP值
            explainer = shap.DeepExplainer(self.model, background_data)
            shap_values = explainer.shap_values(test_samples, check_additivity=False)

            # 3. 可视化1：整体特征重要性条形图（基于所有测试样本）
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values[1],  # 类别1的SHAP值
                test_samples.cpu().numpy(),
                feature_names=all_feature_names,
                plot_type="bar",
                max_display=20,  # 前20个重要特征
                show=False
            )
            plt.title("SHAP特征重要性 (基于所有测试样本, Top 20)", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(config.output_dir, 'SHAP_Summary_Bar.png'), dpi=300)
            plt.close()

            # 4. 可视化2：SHAP值分布点图（基于所有测试样本）
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values[1],
                test_samples.cpu().numpy(),
                feature_names=all_feature_names,
                plot_type="dot",
                max_display=20,
                show=False
            )
            plt.title("SHAP值分布 (基于所有测试样本, Top 20)", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(config.output_dir, 'SHAP_Summary_Dot.png'), dpi=300)
            plt.close()

            ##################################################################
            #5. 可视化3：单个测试样本的决策路径
            # 修复：正确获取基准值
            # 注意：explainer.expected_value 是 NumPy 数组，不是张量
            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[1]
            else:
                # 对于二分类问题，explainer.expected_value 可能是二维数组
                base_value = explainer.expected_value[
                    1] if explainer.expected_value.ndim > 0 else explainer.expected_value

            # 可视化3：单个样本的决策解释
            plt.figure(figsize=(12, 8))
            shap.decision_plot(
                base_value,  # 直接使用基准值，不需要 .cpu().numpy()
                shap_values[1][0],  # 第一个测试样本
                features=test_samples[0].cpu().numpy(),
                feature_names=all_feature_names,
                show=False,
                highlight=0  # 高亮最重要的特征
            )
            plt.title("SHAP决策解释 (单个样本)", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(config.output_dir, 'SHAP_Decision_Plot.png'), dpi=300)
            plt.close()

            # # 5. 可视化3：前5个测试样本的决策路径（关键修改点2：从单个样本改为前5个）
            # # 提取前5个测试样本的SHAP值和特征
            # top5_indices = range(min(5, len(test_samples)))  # 前5个样本
            # top5_shap = shap_values[1][top5_indices]  # 前5个样本的SHAP值
            # top5_features = test_samples[top5_indices].cpu().numpy()
            #
            # # 获取基准值（背景样本的平均预测）
            # if isinstance(explainer.expected_value, list):
            #     base_value = explainer.expected_value[1]
            # else:
            #     base_value = explainer.expected_value[1] if explainer.expected_value.ndim > 0 else explainer.expected_value
            #
            # # 绘制前5个样本的决策路径（并排展示）
            # plt.figure(figsize=(12, 10))
            # shap.decision_plot(
            #     base_value,
            #     top5_shap,  # 前5个样本的SHAP值
            #     features=top5_features,
            #     feature_names=all_feature_names,
            #     show=False,
            #     highlight=0,
            #     legend_labels=[f'样本{i+1}' for i in range(len(top5_indices))]  # 标记每个样本
            # )
            # plt.title("前5个测试样本的SHAP决策路径", fontsize=16)
            # plt.tight_layout()
            # plt.savefig(os.path.join(config.output_dir, 'SHAP_Decision_Plot.png'), dpi=300)
            # plt.close()

            ##################################################################

            # 6. 可视化4：特征类型贡献对比（基于所有测试样本）
            # 计算拉曼和紫外特征的总贡献（关键修改点3：基于所有测试样本）
            total_fl_shap = np.sum(np.abs(shap_values[1][:, :config.input_size_fl]))  # 拉曼总贡献
            total_uv_shap = np.sum(np.abs(shap_values[1][:, config.input_size_fl:]))  # 紫外总贡献

            # 绘制贡献对比图
            plt.figure(figsize=(8, 6))
            labels = ['拉曼特征', '紫外特征']
            values = [total_fl_shap, total_uv_shap]
            colors = ['#1f77b4', '#ff7f0e']
            plt.bar(labels, values, color=colors)
            plt.ylabel('SHAP绝对贡献值', fontsize=12)
            plt.title('拉曼与紫外特征总贡献度对比 (基于所有测试样本)', fontsize=14)
            # 添加数值标签
            for i, v in enumerate(values):
                plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(config.output_dir, 'Feature_Type_Contribution.png'), dpi=300)
            plt.close()

            # 7. 生成SHAP分析报告（基于所有测试样本）
            report = (
                "SHAP贡献度分析报告 (基于所有测试样本):\n"
                f"拉曼特征总贡献度: {total_fl_shap:.4f}\n"
                f"紫外特征总贡献度: {total_uv_shap:.4f}\n"
                f"拉曼/紫外贡献比: {total_fl_shap / total_uv_shap if total_uv_shap > 0 else float('inf'):.2f}:1\n\n"
                "特征重要性排名前10:\n"
            )
            # 计算平均SHAP值并排序
            mean_abs_shap = np.mean(np.abs(shap_values[1]), axis=0)
            sorted_indices = np.argsort(mean_abs_shap)[::-1]
            # 写入前10个特征
            for i, idx in enumerate(sorted_indices[:10]):
                feature_type = "拉曼" if idx < config.input_size_fl else "紫外"
                feature_index = idx if idx < config.input_size_fl else idx - config.input_size_fl
                report += f"{i + 1}. {feature_type}特征_{feature_index} (重要性: {mean_abs_shap[idx]:.4f})\n"
            # 保存报告
            with open(os.path.join(config.output_dir, 'SHAP_Analysis_Report.txt'), 'w') as f:
                f.write(report)
            print(report)

        except Exception as e:
            print(f"SHAP分析失败: {str(e)}")
            import traceback
            traceback.print_exc()


# ==============================
# 6. 执行训练和测试
# ==============================
if __name__ == "__main__":
    trainer = Trainer(config)
    train_acc, train_loss = trainer.train()
    test_metrics, preds, trues, probs = trainer.test()
    trainer.explain_model()