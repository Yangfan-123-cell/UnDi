import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL


class BADGE(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        # 调用父类 AL 的初始化方法，传入配置、模型、未标记数据集、未标记索引、类别数和其他参数
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        # 保存设备信息（CPU 或 GPU）
        self.device = device
        # 初始化预测结果列表，用于存储模型对未标记数据的预测
        self.pred = []

    def get_grad_features(self):
        # 清空预测结果列表，以便重新计算
        self.pred = []
        # 将模型设置为评估模式，避免训练时的梯度计算
        self.model.eval()
        # 根据模型的骨干网络类型确定嵌入维度
        # 如果不是 RN50，则嵌入维度为 512，否则为 1024
        if self.cfg.MODEL.BACKBONE.NAME != "RN50":
            embDim = 512
        else:
            embDim = 1024
        # 获取未标记样本的数量
        num_unlabeled = len(self.U_index)
        # 断言未标记数据集的大小与索引数量一致，确保数据完整性
        assert len(self.unlabeled_set) == num_unlabeled, f"{len(self.unlabeled_dst)} != {num_unlabeled}"
        # 初始化梯度嵌入张量，形状为 [未标记样本数, 嵌入维度 * 类别数]
        grad_embeddings = torch.zeros([num_unlabeled, embDim * self.n_class])
        # 使用无梯度上下文管理器，避免不必要的梯度计算
        with torch.no_grad():
            # 构建未标记数据的 DataLoader，用于批量加载数据
            unlabeled_loader = build_data_loader(
                self.cfg,  # 配置对象
                data_source=self.unlabeled_set,  # 未标记数据集
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  # 批次大小
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,  # 领域数量
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,  # 实例数量
                tfm=build_transform(self.cfg, is_train=False),  # 数据转换（评估模式）
                is_train=False,  # 表示这是评估数据加载器
            )
            # 遍历未标记数据加载器中的每个批次
            for i, batch in enumerate(unlabeled_loader):
                # 将输入图像移动到指定设备
                inputs = batch["img"].to(self.device)
                # 通过模型获取输出和特征，get_feature=True 表示返回特征
                out, features = self.model(inputs, get_feature=True)
                # 对输出应用 softmax 函数，得到每个类别的概率
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                # 获取每个样本预测的最大概率类别
                maxInds = torch.argmax(batchProbs, 1)
                # 将预测结果从 GPU 移到 CPU 并存储到 pred 列表中
                self.pred.append(maxInds.detach().cpu())

                # 对每个样本和每个类别计算梯度嵌入
                for j in range(len(inputs)):
                    for c in range(self.n_class):
                        # 如果当前类别是预测类别
                        if c == maxInds[j]:
                            # 计算梯度嵌入：特征 * (1 - 预测概率)
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                    1 - batchProbs[j][c])
                        else:
                            # 计算梯度嵌入：特征 * (-预测概率)
                            grad_embeddings[i * len(inputs) + j][embDim * c: embDim * (c + 1)] = features[j].clone() * (
                                    -1 * batchProbs[j][c])
        # 将梯度嵌入从 GPU 移到 CPU 并转换为 NumPy 数组返回
        return grad_embeddings.cpu().numpy()

    # kmeans++ 初始化方法，用于选择初始聚类中心
    def k_means_plus_centers(self, X, K):
        # 选择范数最大的点作为第一个中心
        ind = np.argmax([np.linalg.norm(s, 2) for s in X])
        # 初始化中心点列表
        mu = [X[ind]]
        # 存储所有选择的索引
        indsAll = [ind]
        # 初始化中心点归属标记
        centInds = [0.] * len(X)
        # 中心计数器
        cent = 0
        # 打印样本数和总距离的标题
        print('样本数\t总距离')
        # 当中心点数量小于 K 时继续选择
        while len(mu) < K:
            # 如果只有一个中心，计算所有点到该中心的距离
            if len(mu) == 1:
                D2 = pairwise_distances(X, mu).ravel().astype(float)
            else:
                # 计算所有点到最新中心的距离，并更新最小距离
                newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
                for i in range(len(X)):
                    if D2[i] > newD[i]:
                        centInds[i] = cent
                        D2[i] = newD[i]
            # 每选择 100 个中心时打印当前进度
            if len(mu) % 100 == 0:
                print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
            # 如果总距离为 0，进入调试模式
            if sum(D2) == 0.0: pdb.set_trace()
            # 将距离转换为浮点数
            D2 = D2.ravel().astype(float)
            # 根据距离平方归一化，生成概率分布
            Ddist = (D2 ** 2) / sum(D2 ** 2)
            # 使用自定义分布随机选择下一个中心
            customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
            ind = customDist.rvs(size=1)[0]
            # 确保不重复选择已有的中心
            while ind in indsAll: ind = customDist.rvs(size=1)[0]
            # 添加新中心
            mu.append(X[ind])
            indsAll.append(ind)
            cent += 1
        # 返回所有选择的中心索引
        return indsAll

    def select(self, n_query, **kwargs):
        print("n_query的数量(内部,训练集的10%)：",n_query)
        # 获取未标记样本的梯度嵌入
        unlabeled_features = self.get_grad_features()
        # 使用 K-Means++ 选择 n_query 个样本的索引
        selected_indices = self.k_means_plus_centers(X=unlabeled_features, K=n_query)
        # 为每个选择的样本分配分数 1（无实际意义）
        scores = list(np.ones(len(selected_indices)))
        # 将选择的索引映射回原始未标记数据集的索引
        Q_index = [self.U_index[idx] for idx in selected_indices]
        # 返回选择的样本索引
        return Q_index

    def select_by_filter(self, n_query, **kwargs):
        # 获取未标记样本的梯度嵌入
        unlabeled_features = self.get_grad_features()
        # 将预测结果列表拼接为一个张量
        self.pred = torch.cat(self.pred)
        # 初始化预测类别索引列表
        pred_idx = []
        # 初始化返回的索引列表
        ret_idx = []
        # 使用 K-Means++ 选择 10 倍 n_query 个样本的索引
        Q_index = self.k_means_plus_centers(X=unlabeled_features, K=10 * n_query)
        # 从中选择每个类别首次出现的样本
        for q in Q_index:
            if int(self.pred[q]) not in pred_idx:
                pred_idx.append(int(self.pred[q]))
                ret_idx.append(q)
        # 如果已经覆盖所有类别
        if len(pred_idx) == self.n_class:
            # 将选择的索引映射回原始未标记数据集的索引
            ret_idx = [self.U_index[idx] for idx in ret_idx]
            # 打印所有类别的预测索引
            print(f"预测索引 (all the classes): {pred_idx}")
            # 返回选择的样本索引和 None
            return ret_idx, None
        # 如果未能覆盖所有类别，打印失败信息
        print("Fail to get all the classes!!!")
        # 继续从剩余样本中选择，直到覆盖所有类别
        for q in Q_index:
            if len(ret_idx) == self.n_class:
                ret_idx = [self.U_index[idx] for idx in ret_idx]
                print(f"pred idx: {pred_idx}")
                return ret_idx, None
            if q not in ret_idx:
                pred_idx.append(int(self.pred[q]))
                ret_idx.append(q)
        # 如果仍未成功，选择失败，抛出环境错误
        raise EnvironmentError
                
                