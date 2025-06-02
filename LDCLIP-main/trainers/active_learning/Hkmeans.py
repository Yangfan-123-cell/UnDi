import torch
import numpy as np
import pandas as pd
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from sklearn.cluster import KMeans
import time
from .AL import AL


class Hkmeans(AL):
    """基于熵加权 K-means 聚类的主动学习方法，继承自 AL 类"""

    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        """
        初始化 Hkmeans 类
        参数:
            cfg: 配置文件对象，包含数据加载和模型参数
            model: 当前训练的神经网络模型
            unlabeled_dst: 未标记数据集，包含未标记样本
            U_index: 未标记样本的全局索引列表
            n_class: 数据集的类别数
            device: 计算设备（如 'cuda' 或 'cpu'）
            **kwargs: 其他可选参数，传递给父类
        """
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self._uncertainty_scores = None  # 保存最近计算的熵值

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

            print("| 计算未标记集的熵值")
            for i, data in enumerate(selection_loader):
                inputs = data["img"].to(self.device)  # 移动输入到设备
                if "label" in data:
                    batch_labels = data["label"].cpu().numpy()
                    labels = np.append(labels, batch_labels)

                preds = self.model(inputs, get_feature=False)  # 获取预测
                preds = torch.nn.functional.softmax(preds, dim=1).cpu().numpy()  # 转换为概率
                # 计算熵：H = -∑(p * log(p))，添加 1e-6 避免 log(0)
                entropy_batch = -(np.log(preds + 1e-6) * preds).sum(axis=1)
                entropies = np.append(entropies, entropy_batch)

            # 打印未归一化的熵值统计信息
            print(f"| 未归一化熵值统计：均值={np.mean(entropies):.4f}, 标准差={np.std(entropies):.4f}, "
                  f"最小值={np.min(entropies):.4f}, 最大值={np.max(entropies):.4f}")

        # 保存熵和标签信息以便后续使用
        self._uncertainty_scores = entropies
        self._sample_labels = labels if len(labels) > 0 else None

        return entropies

    def get_features(self):
        """
        获取所有未标记样本的特征表示，用于 K-means 聚类

        返回:
            features: 样本的特征矩阵
        """
        self.model.eval()
        features = []
        with torch.no_grad():
            loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False
            )
            for data in loader:
                inputs = data["img"].to(self.device)
                feats = self.model(inputs, get_feature=True)[1]  # 获取特征
                features.append(feats.cpu().numpy())
        return np.vstack(features)

    def get_sample_labels(self):
        """获取样本标签（如果有的话）"""
        if not hasattr(self, '_sample_labels') or self._sample_labels is None:
            labels = []
            for sample in self.unlabeled_set:
                if hasattr(sample, 'label'):
                    labels.append(sample.label.item() if isinstance(sample.label, torch.Tensor) else sample.label)
                else:
                    labels.append(-1)  # 如果没有标签则用-1表示
            self._sample_labels = np.array(labels)
        return self._sample_labels

    def select(self, n_cand, **kwargs):
        # 计算熵值
        entropies = self.rank_uncertainty()
        self._uncertainty_scores = entropies

        # 获取样本特征
        print("| 提取未标记样本的特征")
        features = self.get_features()

        # 使用归一化后的熵值作为权重进行加权 K-means 聚类
        n_query = self.n_class  # 聚类数量为类别数
        samples_per_cluster = 5  # 每个簇选择5个样本
        print(f"| 执行加权 K-means 聚类，簇数={n_query}，每个簇选择样本数={samples_per_cluster}")

        # 归一化熵值为正权重（用于加权）
        weights = entropies / (np.max(entropies) + 1e-6)  # 归一化到 (0,1]
        weighted_features = features * weights[:, np.newaxis]  # 按归一化熵值加权特征

        # 执行 K-means 聚类
        kmeans = KMeans(n_clusters=n_query, random_state=0, n_init=10).fit(weighted_features)
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_

        print(f"| 已完成聚类，共 {n_query} 个簇")

        # 从每个簇中选择五个最接近中心的样本
        selection_result = []
        for cluster_idx in range(n_query):
            cluster_samples = np.where(cluster_labels == cluster_idx)[0]
            if len(cluster_samples) == 0:
                print(f"| 警告：簇 {cluster_idx} 为空，跳过")
                continue

            # 计算到簇中心的距离（基于原始特征）
            cluster_center = cluster_centers[cluster_idx]
            distances = np.linalg.norm(
                features[cluster_samples] - cluster_center,
                axis=1
            )

            # 按距离排序
            sorted_indices = np.argsort(distances)

            # 选择最接近中心的5个样本，或者簇中的所有样本（如果少于5个）
            num_to_select = min(samples_per_cluster, len(cluster_samples))
            selected_from_cluster = [cluster_samples[idx] for idx in sorted_indices[:num_to_select]]

            # 将这些样本添加到结果中
            for idx in selected_from_cluster:
                selection_result.append(idx)

            # 打印这些样本的未归一化熵值
            print(f"| 簇 {cluster_idx} 选择的 {num_to_select} 个样本的未归一化熵值:")
            for i, idx in enumerate(selected_from_cluster):
                global_idx = self.U_index[idx]
                entropy_value = self._uncertainty_scores[idx]
                print(f"|   样本 {i + 1}: 局部索引 {idx}, 全局索引 {global_idx}, 熵值 {entropy_value:.4f}")

        # 如果选择的样本数小于 n_query * samples_per_cluster，随机补充
        target_count = n_query * samples_per_cluster
        if len(selection_result) < target_count:
            print(f"| 警告：只选择了 {len(selection_result)} 个样本，少于目标 {target_count}，将随机补充")
            remaining = list(set(range(len(self.unlabeled_set))) - set(selection_result))
            np.random.shuffle(remaining)
            additional = remaining[:target_count - len(selection_result)]
            selection_result.extend(additional)

            # 打印补充的样本的熵值
            print(f"| 随机补充的 {len(additional)} 个样本的未归一化熵值:")
            for i, idx in enumerate(additional):
                global_idx = self.U_index[idx]
                entropy_value = self._uncertainty_scores[idx]
                print(f"|   补充样本 {i + 1}: 局部索引 {idx}, 全局索引 {global_idx}, 熵值 {entropy_value:.4f}")

        print(f"| 最终选择了 {len(selection_result)} 个样本")

        # 获取样本标签（如果有）
        sample_labels = self.get_sample_labels()

        # 构造选择样本信息
        selection_data = {
            "local_index": selection_result,
            "global_index": [self.U_index[idx] for idx in selection_result],
            "entropy": [float(self._uncertainty_scores[idx]) for idx in selection_result],  # 未归一化的熵值
            "label": [sample_labels[idx] for idx in selection_result]
        }

        # 返回全局索引
        Q_index = selection_data["global_index"]
        return Q_index