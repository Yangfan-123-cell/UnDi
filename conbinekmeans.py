import torch
import numpy as np
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.cluster import KMeans

from .AL import AL


class Learnability(AL):
    """基于多轮训练置信度和变异性的样本选择类，继承自 AL 类，结合熵和多样性"""

    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, num_epochs=10, **kwargs):
        """
        初始化 Learnability 类
        参数:
            cfg: 配置文件对象，包含数据加载和模型参数
            model: 当前训练的神经网络模型
            unlabeled_dst: 未标记数据集，包含未标记样本
            U_index: 未标记样本的全局索引列表
            n_class: 数据集的类别数
            device: 计算设备（如 'cuda' 或 'cpu'）
            num_epochs: 用于计算置信度和变异性的训练周期数
            **kwargs: 其他可选参数，传递给父类
        """
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.num_epochs = num_epochs  # 训练周期数

    def rank_uncertainty(self):
        """
        计算未标记样本的熵，作为不确定性得分

        返回:
            entropies: 每个未标记样本的熵值
        """
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            selection_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            entropies = np.array([])  # 初始化熵得分数组

            print("| Calculating uncertainty of Unlabeled set")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)  # 移动输入到设备
                preds = self.model(inputs, get_feature=False)  # 获取预测
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()  # 转换为概率
                # 计算熵：H = -∑(p * log(p))，添加 1e-6 避免 log(0)
                entropy_batch = -(np.log(preds + 1e-6) * preds).sum(axis=1)
                entropies = np.append(entropies, entropy_batch)

        return entropies

    def rank_learnability(self):
        """
        计算未标记样本的置信度和变异性

        返回:
            confidences: 每个未标记样本的置信度（预测最大概率的均值）
            variabilities: 每个未标记样本的变异性（预测最大概率的标准差）
        """
        self.model.eval()  # 设置模型为评估模式
        selection_loader = build_data_loader(
            self.cfg,
            data_source=self.unlabeled_set,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=build_transform(self.cfg, is_train=False),
            is_train=False,
        )

        all_epoch_preds = []  # 存储每个样本在每个周期的预测概率
        print("| 开始多轮预测以计算未标记集的置信度和变异性")
        for epoch in range(self.num_epochs):
            epoch_preds = []  # 当前周期的预测概率
            with torch.no_grad():
                for i, data in enumerate(selection_loader):
                    inputs = data["img"].to(self.device)
                    preds = self.model(inputs, get_feature=False)
                    preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()
                    epoch_preds.append(preds)
            epoch_preds = np.vstack(epoch_preds)  # 合并为 [num_samples, num_classes]
            all_epoch_preds.append(epoch_preds)
            print(f"| 完成第 {epoch + 1}/{self.num_epochs} 轮预测")

        # 转换为 NumPy 数组，形状: [num_epochs, num_samples, num_classes]
        all_epoch_preds = np.array(all_epoch_preds)

        # 计算置信度和变异性
        num_samples = all_epoch_preds.shape[1]
        confidences = np.zeros(num_samples)
        variabilities = np.zeros(num_samples)

        for i in range(num_samples):
            sample_preds = all_epoch_preds[:, i, :]  # 形状: [num_epochs, num_classes]
            max_prob_indices = np.argmax(sample_preds.mean(axis=0))  # 平均概率的最大类别
            max_prob_sequence = sample_preds[:, max_prob_indices]  # 最大类别的概率序列
            confidences[i] = np.mean(max_prob_sequence)  # 置信度
            variabilities[i] = np.std(max_prob_sequence)  # 变异性

        return confidences, variabilities

    def get_features(self, indices):
        """
        获取指定样本的特征表示，用于多样性选择

        参数:
            indices: 样本索引列表

        返回:
            features: 样本的特征矩阵
        """
        self.model.eval()
        features = []
        with torch.no_grad():
            loader = build_data_loader(
                self.cfg,
                data_source=[self.unlabeled_set[i] for i in indices],
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False
            )
            for data in loader:
                inputs = data["img"].to(self.device)
                feats = self.model(inputs, get_feature=True)[1]
                features.append(feats.cpu().numpy())
        return np.vstack(features)

    def run(self, n_cand):
        """
        执行样本选择过程，结合熵和变异性，并考虑样本多样性

        参数:
            n_cand: 候选样本数量（通常为数据集大小的一个百分比，如10%）

        返回:
            selection_result: 选择的样本索引（相对于未标记集）
            scores: 所有未标记样本的可学习性得分
        """
        # 获取类别数作为n_query
        n_query = self.n_class
        print(f"| 类别数量 n_query: {n_query}")
        print(f"| 候选样本数量 n_cand: {n_cand}")

        # 计算置信度、变异性和熵
        confidences, variabilities = self.rank_learnability()
        entropies = self.rank_uncertainty()

        # 综合得分：变异性 + 熵，调整权重
        alpha = 0.2
        scores = alpha * variabilities / (1 + confidences + 1e-6) + (1 - alpha) * entropies

        # 选择前n_cand个得分最高的样本作为候选样本
        candidate_indices = np.argsort(scores)[-n_cand:]
        print(f"| 已选择 {len(candidate_indices)} 个候选样本")

        # 获取候选样本的特征
        candidate_features = self.get_features(candidate_indices)

        # 对候选样本进行K-means聚类，K=n_query（类别数）
        kmeans = KMeans(n_clusters=n_query, random_state=0, n_init=10).fit(candidate_features)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        print(f"| 已完成聚类，共 {n_query} 个簇")

        # 从每个簇中选择最接近中心的样本
        selection_result = []
        for cluster_idx in range(n_query):
            # 获取当前簇的所有样本索引
            cluster_samples = np.where(cluster_labels == cluster_idx)[0]

            if len(cluster_samples) == 0:
                continue

            # 计算到簇中心的距离
            cluster_center = cluster_centers[cluster_idx]
            distances = np.linalg.norm(
                candidate_features[cluster_samples] - cluster_center,
                axis=1
            )

            # 选择最接近中心的样本
            closest_idx = cluster_samples[np.argmin(distances)]
            selection_result.append(candidate_indices[closest_idx])

        # 如果选择的样本数小于n_query，随机补充
        if len(selection_result) < n_query:
            print(f"| 警告：只选择了 {len(selection_result)} 个样本，少于目标 {n_query}，将随机补充")
            remaining = list(set(candidate_indices) - set(selection_result))
            np.random.shuffle(remaining)
            selection_result.extend(remaining[:n_query - len(selection_result)])

        print(f"| 最终选择了 {len(selection_result)} 个样本")
        return selection_result, scores

    def select(self, n_cand, **kwargs):
        """
        选择指定数量的候选样本并返回其全局索引
        参数:
            n_cand: 候选样本数量（通常为数据集大小的一个百分比，如10%）
        返回:
            Q_index: 选择的样本在全局数据集中的索引
        """
        selected_indices, scores = self.run(n_cand)


        # 转换为全局索引
        Q_index = [self.U_index[idx] for idx in selected_indices]
        return Q_index