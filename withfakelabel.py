import torch
import numpy as np
import pandas as pd
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.cluster import KMeans

from .AL import AL


class Learnability(AL):
    """基于伪标签、多轮训练置信度和变异性的样本选择类，继承自 AL 类"""

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
        self._uncertainty_scores = None  # 保存最近计算的不确定性得分
        self.pseudo_threshold = 0.99  # 伪标签的置信度阈值

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
            labels = np.array([])  # 初始化标签数组
            all_preds = np.array([])  # 初始化预测概率数组
            all_pred_labels = np.array([])  # 初始化预测标签数组

            print("| 计算不确定和未标签数据")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)  # 移动输入到设备
                if "label" in data:
                    batch_labels = data["label"].cpu().numpy()
                    labels = np.append(labels, batch_labels)

                preds = self.model(inputs, get_feature=False)  # 获取预测
                preds_softmax = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()  # 转换为概率

                # 保存预测的类别和最大概率
                pred_labels = np.argmax(preds_softmax, axis=1)
                pred_max_probs = np.max(preds_softmax, axis=1)

                if len(all_preds) == 0:
                    all_preds = preds_softmax
                    all_pred_labels = pred_labels
                else:
                    all_preds = np.vstack((all_preds, preds_softmax))
                    all_pred_labels = np.append(all_pred_labels, pred_labels)

                # 计算熵：H = -∑(p * log(p))，添加 1e-6 避免 log(0)
                entropy_batch = -(np.log(preds_softmax + 1e-6) * preds_softmax).sum(axis=1)
                entropies = np.append(entropies, entropy_batch)

        # 保存熵、标签、预测概率和最大概率预测标签信息以便后续使用
        self._uncertainty_scores = entropies
        self._sample_labels = labels if len(labels) > 0 else None
        self._pred_probs = all_preds  # 预测概率
        self._pred_labels = all_pred_labels  # 预测的标签
        self._pred_max_probs = np.max(all_preds, axis=1)  # 最大预测概率

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

    def get_sample_labels(self):
        """获取样本标签（如果有的话）"""
        if not hasattr(self, '_sample_labels') or self._sample_labels is None:
            # 尝试从 self.unlabeled_set 中提取标签
            labels = []
            for sample in self.unlabeled_set:
                if hasattr(sample, 'label'):
                    labels.append(sample.label.item() if isinstance(sample.label, torch.Tensor) else sample.label)
                else:
                    labels.append(-1)  # 如果没有标签则用-1表示
            self._sample_labels = np.array(labels)
        return self._sample_labels

    def assign_pseudo_labels(self):
        """
        为高置信度样本分配伪标签

        返回:
            pseudo_indices: 被分配伪标签的样本索引
            pseudo_labels: 对应的伪标签
        """
        if not hasattr(self, '_pred_max_probs'):
            # 需要先计算一次预测
            self.rank_uncertainty()

        # 筛选出置信度大于阈值的样本
        high_conf_indices = np.where(self._pred_max_probs >= self.pseudo_threshold)[0]
        pseudo_labels = self._pred_labels[high_conf_indices].astype(int)

        print(f"| 发现 {len(high_conf_indices)} 个高置信度样本 (阈值: {self.pseudo_threshold})")

        return high_conf_indices, pseudo_labels

    def run(self, n_cand):
        """
        执行样本选择过程:
        1. 先选择高置信度样本分配伪标签
        2. 剩余样本进行变异性和熵的综合评分
        3. 使用K-means聚类，每个簇选择一个样本

        参数:
            n_cand: 候选样本数量

        返回:
            selection_result: 选择的样本索引（相对于未标记集）
            scores: 所有未标记样本的可学习性得分
        """
        # 获取类别数作为聚类数
        n_query = self.n_class
        # 每个簇只选择1个样本
        samples_per_cluster = 1
        print(f"| 类别数量 n_query: {n_query}")
        print(f"| 候选样本数量 n_cand: {n_cand}")
        print(f"| 每个簇选择样本数: {samples_per_cluster}")

        # 1. 计算预测和不确定性得分
        if not hasattr(self, '_uncertainty_scores') or self._uncertainty_scores is None:
            self.rank_uncertainty()

        # 2. 找出高置信度样本并分配伪标签
        pseudo_indices, pseudo_labels = self.assign_pseudo_labels()

        # 记录结果
        pseudo_selection = {
            "indices": pseudo_indices.tolist(),
            "labels": pseudo_labels.tolist(),
            "probs": self._pred_max_probs[pseudo_indices].tolist()
        }

        #print(f"| 已选择 {len(pseudo_indices)} 个高置信度样本作为伪标签样本")

        # 3. 剩余未标记样本中计算置信度和变异性
        # 创建剩余样本的掩码
        mask = np.ones(len(self.unlabeled_set), dtype=bool)
        mask[pseudo_indices] = False
        remaining_indices = np.where(mask)[0]

        if len(remaining_indices) > 0:
            #print(f"| 剩余 {len(remaining_indices)} 个样本进行学习性评估")

            # 计算剩余样本的置信度和变异性
            confidences, variabilities = self.rank_learnability()

            # 综合得分：变异性 + 熵，调整权重
            alpha = 0.3
            scores = np.zeros_like(self._uncertainty_scores)
            scores[remaining_indices] = (
                    alpha * variabilities[remaining_indices] / (1 + confidences[remaining_indices] + 1e-6) +
                    (1 - alpha) * self._uncertainty_scores[remaining_indices]
            )

            # 选择得分最高的样本作为候选
            remaining_sorted = remaining_indices[np.argsort(scores[remaining_indices])[::-1]]
            candidate_indices = remaining_sorted[:n_cand - len(pseudo_indices)]

            print(f"| 从剩余样本中选择 {len(candidate_indices)} 个候选样本")

            if len(candidate_indices) > 0:
                # 获取候选样本的特征
                candidate_features = self.get_features(candidate_indices)

                # 对候选样本进行K-means聚类
                n_clusters = min(n_query, len(candidate_indices))
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(candidate_features)
                cluster_centers = kmeans.cluster_centers_
                cluster_labels = kmeans.labels_

                print(f"| 已完成聚类，共 {n_clusters} 个簇")

                # 从每个簇中选择一个样本（最接近中心的）
                cluster_selections = []
                for cluster_idx in range(n_clusters):
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
                    cluster_selections.append(candidate_indices[closest_idx])

                    # 打印该样本的信息
                    global_idx = self.U_index[candidate_indices[closest_idx]]
                    entropy_value = self._uncertainty_scores[candidate_indices[closest_idx]]
                    #print(f"|   簇 {cluster_idx}: 选择局部索引 {candidate_indices[closest_idx]}, "
                          #f"全局索引 {global_idx}, 熵值 {entropy_value:.4f}")

                # 合并伪标签样本和聚类选择的样本
                selection_result = list(pseudo_indices) + cluster_selections
            else:
                # 如果所有样本都用于伪标签，则只返回伪标签样本
                selection_result = list(pseudo_indices)
        else:
            # 如果所有样本都用于伪标签，则只返回伪标签样本
            selection_result = list(pseudo_indices)

        print(f"| 最终选择了 {len(selection_result)} 个样本，其中 {len(pseudo_indices)} 个伪标签样本")

        # 保存伪标签信息，以便在select方法中返回
        self._pseudo_selection = pseudo_selection

        return selection_result, self._uncertainty_scores

    def select(self, n_cand, **kwargs):
        """
        选择指定数量的候选样本并返回其全局索引
        参数:
            n_cand: 候选样本数量（通常为数据集大小的一个百分比，如10%）
        返回:
            Q_index: 选择的样本在全局数据集中的索引
        """
        # 每次 select 前都重新计算不确定性，确保与当前 unlabeled_set 同步
        self._uncertainty_scores = self.rank_uncertainty()

        # 获取置信度、变异性、聚类等综合策略选样
        selected_indices, scores = self.run(n_cand)

        # 获取样本标签（如果有）
        sample_labels = self.get_sample_labels()

        # 处理伪标签样本（如果在run中有生成）
        if hasattr(self, '_pseudo_selection'):
            # 为伪标签样本添加标记，以便后续处理
            pseudo_samples = set(self._pseudo_selection["indices"])
            pseudo_labels_map = dict(zip(self._pseudo_selection["indices"], self._pseudo_selection["labels"]))
        else:
            pseudo_samples = set()
            pseudo_labels_map = {}

        # 构造选择样本信息
        selection_data = {
            "local_index": selected_indices,
            "global_index": [self.U_index[idx] for idx in selected_indices],
            "entropy": [float(scores[idx]) for idx in selected_indices],
            "label": [
                pseudo_labels_map[idx] if idx in pseudo_samples
                else sample_labels[idx] for idx in selected_indices
            ],
            "is_pseudo": [idx in pseudo_samples for idx in selected_indices]
        }

        # 返回全局索引
        Q_index = selection_data["global_index"]
        return Q_index