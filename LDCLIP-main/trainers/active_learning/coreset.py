from .AL import AL
import torch
import numpy as np

from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader


class Coreset(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, val_set, n_class, **kwargs):
        # 初始化Coreset类，继承自AL基类。
        # cfg: 配置文件，包含超参数、数据加载配置等
        # model: 训练模型
        # unlabeled_dst: 未标注数据集
        # U_index: 未标注数据集的索引
        # val_set: 验证集，用作初始标注集
        # n_class: 类别数
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)

        # 将验证集作为当前的标注数据集
        self.labeled_in_set = val_set

    def get_features(self):
        """
        通过前向传播从模型中获取已标注和未标注数据的特征。
        """
        self.model.eval()  # 将模型设置为评估模式
        labeled_features, unlabeled_features = None, None

        with torch.no_grad():  # 不需要计算梯度，加速计算过程
            # 生成数据加载器，用于加载已标注数据集
            labeled_in_loader = build_data_loader(
                self.cfg,
                data_source=self.labeled_in_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            # 生成数据加载器，用于加载未标注数据集
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            # 获取已标注数据集的特征
            for data in labeled_in_loader:
                inputs = data["img"].cuda()  # 将数据移动到GPU
                out, features = self.model(inputs, get_feature=True)  # 获取模型输出和特征
                if labeled_features is None:
                    labeled_features = features  # 第一次获取特征
                else:
                    labeled_features = torch.cat((labeled_features, features), 0)  # 拼接特征

            # 获取未标注数据集的特征
            for data in unlabeled_loader:
                inputs = data["img"].cuda()  # 将数据移动到GPU
                out, features = self.model(inputs, get_feature=True)  # 获取模型输出和特征
                if unlabeled_features is None:
                    unlabeled_features = features  # 第一次获取特征
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)  # 拼接特征

        return labeled_features, unlabeled_features  # 返回已标注和未标注的特征

    def k_center_greedy(self, labeled, unlabeled, n_query):
        """
        使用k-center greedy策略选择最具代表性的样本。
        通过计算已标注集和未标注集之间的最小距离，选择最远的样本。
        """
        # 计算已标注样本与未标注样本之间的距离，选择最小距离（即与最近已标注样本的距离）
        min_dist = torch.min(torch.cdist(labeled[0:2, :], unlabeled), 0).values

        # 迭代计算已标注样本与未标注样本之间的最小距离（分批处理，避免内存问题）
        for j in range(2, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist_matrix = torch.cdist(labeled[j:j + 100, :], unlabeled)  # 计算当前批次的距离矩阵
            else:
                dist_matrix = torch.cdist(labeled[j:, :], unlabeled)  # 处理最后一批
            min_dist = torch.stack((min_dist, torch.min(dist_matrix, 0).values))  # 更新最小距离
            min_dist = torch.min(min_dist, 0).values  # 取最小的距离值

        min_dist = min_dist.reshape((1, min_dist.size(0)))  # 重塑最小距离的形状
        farthest = torch.argmax(min_dist)  # 找到最远的样本索引

        # 初始化选择的样本列表，将最远的样本添加到选择的样本中
        greedy_indices = torch.tensor([farthest])

        # 继续选择最远的n_query个样本
        for i in range(n_query - 1):
            # 计算选择的样本与未标注样本之间的距离
            dist_matrix = torch.cdist(unlabeled[greedy_indices[-1], :].reshape((1, -1)), unlabeled)
            min_dist = torch.stack((min_dist, dist_matrix))  # 更新最小距离
            min_dist = torch.min(min_dist, 0).values  # 取最小的距离值

            farthest = torch.tensor([torch.argmax(min_dist)])  # 找到最远的样本
            greedy_indices = torch.cat((greedy_indices, farthest), 0)  # 将最远样本添加到选择列表中

        return greedy_indices.cpu().numpy()  # 返回选择的样本索引

    def select(self, n_query, **kwargs):
        """
        根据k-center greedy策略从未标注数据集中选择n_query个样本。
        """
        labeled_features, unlabeled_features = self.get_features()  # 获取已标注和未标注数据的特征
        selected_indices = self.k_center_greedy(labeled_features, unlabeled_features, n_query)  # 选择样本索引
        scores = list(np.ones(len(selected_indices)))  # 目前没有评分机制，所有样本分数设为1

        # 将选择的索引映射回原始未标注数据集的索引
        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index  # 返回选择的样本索引
