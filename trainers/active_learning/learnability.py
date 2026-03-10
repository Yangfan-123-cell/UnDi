import torch
import numpy as np
import pandas as pd
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.cluster import KMeans

from .AL import AL


class Learnability(AL):

    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, num_epochs=10, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.num_epochs = num_epochs
        self._uncertainty_scores = None 
        self.pseudo_threshold = 0.99
        self.high_entropy_threshold = 0.8
        self.low_entropy_threshold = 0.4
        self.samples_per_cluster = 3

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
                is_train=True,
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

        # 计算归一化熵
        log_c = np.log(self.n_class)
        normalized_entropies = entropies / log_c

        self._uncertainty_scores = entropies
        self._normalized_entropies = normalized_entropies
        self._sample_labels = labels if len(labels) > 0 else None
        self._pred_probs = all_preds  # 预测概率
        self._pred_labels = all_pred_labels  # 预测的标签
        self._pred_max_probs = np.max(all_preds, axis=1)  # 最大预测概率

        return entropies

    def rank_learnability(self):
        self.model.eval()
        selection_loader = build_data_loader(
            self.cfg,
            data_source=self.unlabeled_set,
            batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
            n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
            tfm=build_transform(self.cfg, is_train=False),
            is_train=False,
        )

        all_epoch_preds = []
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

        if not hasattr(self, '_pred_max_probs'):
            # 需要先计算一次预测
            self.rank_uncertainty()

        # 筛选出置信度大于阈值的样本
        high_conf_indices = np.where(self._pred_max_probs >= self.pseudo_threshold)[0]
        pseudo_labels = self._pred_labels[high_conf_indices].astype(int)

        print(f"| 发现 {len(high_conf_indices)} 个高置信度样本 (阈值: {self.pseudo_threshold})")

        return high_conf_indices, pseudo_labels

    def run(self, n_cand):

        n_query = self.n_class
        samples_per_cluster = self.samples_per_cluster
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

        # 3. 剩余未标记样本中计算置信度和变异性
        # 创建剩余样本的掩码
        mask = np.ones(len(self.unlabeled_set), dtype=bool)
        mask[pseudo_indices] = False
        remaining_indices = np.where(mask)[0]

        # 熵统计结果初始化
        entropy_stats = {
            'high_entropy_count': 0,  # 高熵样本计数
            'low_entropy_count': 0,  # 低熵样本计数
            'medium_entropy_count': 0  # 中等熵样本计数
        }

        if len(remaining_indices) > 0:
            print(f"| 剩余 {len(remaining_indices)} 个样本进行学习性评估")

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

                # 从每个簇中选择top5样本（最接近中心的）
                cluster_selections = []

                # 各熵级别样本统计
                cluster_entropy_stats = []

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

                    # 按距离排序，选择最近的samples_per_cluster个样本
                    sorted_indices = np.argsort(distances)
                    top_k = min(samples_per_cluster, len(sorted_indices))
                    closest_indices = sorted_indices[:top_k]

                    # 当前簇选择的样本索引
                    selected_cluster_samples = [candidate_indices[cluster_samples[idx]] for idx in closest_indices]
                    cluster_selections.extend(selected_cluster_samples)

                    # 统计选中样本中的高熵和低熵样本
                    cluster_high_entropy = 0
                    cluster_low_entropy = 0
                    cluster_medium_entropy = 0

                    for sample_idx in selected_cluster_samples:
                        norm_entropy = self._normalized_entropies[sample_idx]
                        if norm_entropy > self.high_entropy_threshold:
                            cluster_high_entropy += 1
                            entropy_stats['high_entropy_count'] += 1
                        elif norm_entropy < self.low_entropy_threshold:
                            cluster_low_entropy += 1
                            entropy_stats['low_entropy_count'] += 1
                        else:
                            cluster_medium_entropy += 1
                            entropy_stats['medium_entropy_count'] += 1

                    # 记录当前簇的统计信息
                    cluster_entropy_stats.append({
                        'cluster_id': cluster_idx,
                        'total_samples': len(selected_cluster_samples),
                        'high_entropy': cluster_high_entropy,
                        'low_entropy': cluster_low_entropy,
                        'medium_entropy': cluster_medium_entropy
                    })

                    print(f"|   簇 {cluster_idx}: 选择 {len(selected_cluster_samples)} 样本, "
                          f"高熵 {cluster_high_entropy}, 低熵 {cluster_low_entropy}, 中等熵 {cluster_medium_entropy}")

                # 合并伪标签样本和聚类选择的样本
                selection_result = list(pseudo_indices) + cluster_selections

                pseudo_high_entropy = 0
                pseudo_low_entropy = 0
                pseudo_medium_entropy = 0

                for idx in pseudo_indices:
                    norm_entropy = self._normalized_entropies[idx]
                    if norm_entropy > self.high_entropy_threshold:
                        pseudo_high_entropy += 1
                    elif norm_entropy < self.low_entropy_threshold:
                        pseudo_low_entropy += 1
                    else:
                        pseudo_medium_entropy += 1

                print(
                    f"| 伪标签样本熵分布: 高熵 {pseudo_high_entropy}, 低熵 {pseudo_low_entropy}, 中等熵 {pseudo_medium_entropy}")
                print(f"| 聚类选择样本熵分布: 高熵 {entropy_stats['high_entropy_count']}, "
                      f"低熵 {entropy_stats['low_entropy_count']}, 中等熵 {entropy_stats['medium_entropy_count']}")
                self._cluster_entropy_stats = cluster_entropy_stats
            else:
                selection_result = list(pseudo_indices)
        else:
            selection_result = list(pseudo_indices)

        print(f"| 最终选择了 {len(selection_result)} 个样本，其中 {len(pseudo_indices)} 个伪标签样本")

        self._pseudo_selection = pseudo_selection
        self._entropy_stats = entropy_stats

        return selection_result, self._uncertainty_scores

    def select(self, n_cand, **kwargs):
        self._uncertainty_scores = self.rank_uncertainty()
        selected_indices, scores = self.run(n_cand)
        sample_labels = self.get_sample_labels()
        if hasattr(self, '_pseudo_selection'):
            pseudo_samples = set(self._pseudo_selection["indices"])
            pseudo_labels_map = dict(zip(self._pseudo_selection["indices"], self._pseudo_selection["labels"]))
        else:
            pseudo_samples = set()
            pseudo_labels_map = {}

        selection_data = {
            "local_index": selected_indices,
            "global_index": [self.U_index[idx] for idx in selected_indices],
            "entropy": [float(scores[idx]) for idx in selected_indices],
            "normalized_entropy": [float(self._normalized_entropies[idx]) for idx in selected_indices],
            "label": [
                pseudo_labels_map[idx] if idx in pseudo_samples
                else sample_labels[idx] for idx in selected_indices
            ],
            "is_pseudo": [idx in pseudo_samples for idx in selected_indices]

        if hasattr(self, '_entropy_stats'):
            selection_data["entropy_stats"] = self._entropy_stats

        if hasattr(self, '_cluster_entropy_stats'):
            selection_data["cluster_stats"] = self._cluster_entropy_stats

        Q_index = selection_data["global_index"]
        return Q_index
