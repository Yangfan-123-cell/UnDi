import torch
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
import random

from .AL import AL


class PCB(AL):  # 继承自主动学习类AL
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, statistics, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)  # 调用父类的初始化方法
        self.device = device  # 设置设备（CPU或GPU）
        self.pred = []  # 初始化一个列表，用来存储预测结果
        self.statistics = statistics  # 类别的统计信息，通常是一个记录每个类别已标注样本数的列表或张量

    def select(self, n_query, **kwargs):
        self.pred = []  # 清空上次的预测结果
        self.model.eval()  # 设置模型为评估模式（不计算梯度）

        num_unlabeled = len(self.U_index)  # 获取未标注数据集的大小
        # 检查未标注数据集的大小是否与U_index一致
        assert len(self.unlabeled_set) == num_unlabeled, f"{len(self.unlabeled_dst)} != {num_unlabeled}"

        with torch.no_grad():  # 不需要计算梯度，节省内存和计算资源
            unlabeled_loader = build_data_loader(
                self.cfg,  # 配置文件
                data_source=self.unlabeled_set,  # 未标注数据集
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,  # 批次大小
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,  # 域数
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,  # 每个batch的样本数
                tfm=build_transform(self.cfg, is_train=False),  # 图像预处理
                is_train=False,  # 是否为训练模式
            )

            # 生成整个未标注样本的特征集
            for i, batch in enumerate(unlabeled_loader):
                inputs = batch["img"].to(self.device)  # 将输入图像移动到指定设备
                out, features = self.model(inputs, get_feature=True)  # 获取模型输出和特征
                batchProbs = torch.nn.functional.softmax(out, dim=1).data  # 计算每个类的概率
                maxInds = torch.argmax(batchProbs, 1)  # 获取每个样本的最大概率类别（预测标签）
                self.pred.append(maxInds.detach().cpu())  # 将每个样本的最大类别概率保存
        self.pred = torch.cat(self.pred)  # 合并所有批次的预测结果

        Q_index = []  # 用来存储选中的待标注样本索引

        # 开始选择样本，直到选满n_query个样本
        while len(Q_index) < n_query:
            min_cls = int(torch.argmin(self.statistics))  # 找到类别中已标注样本数最少的类别
            sub_pred = (self.pred == min_cls).nonzero().squeeze(dim=1).tolist()  # 获取所有预测为该类别的样本索引

            # 如果该类别没有样本，随机选择一个未标注样本
            if len(sub_pred) == 0:
                num = random.randint(0, num_unlabeled - 1)
                while num in Q_index:  # 确保选择的样本不在已经选择的列表中
                    num = random.randint(0, num_unlabeled - 1)
                Q_index.append(num)  # 添加选择的样本索引
            else:
                # 如果有该类别的预测样本，打乱顺序，按顺序选择未标注样本
                random.shuffle(sub_pred)
                for idx in sub_pred:
                    if idx not in Q_index:  # 如果该样本未被选择
                        Q_index.append(idx)  # 添加该样本索引
                        self.statistics[min_cls] += 1  # 更新该类别的已标注样本数量
                        break  # 选择该样本后跳出
                else:  # 如果该类别的所有样本都已经选择
                    # 如果没有可以选择的样本，则随机选择一个样本
                    num = random.randint(0, num_unlabeled - 1)
                    while num in Q_index:
                        num = random.randint(0, num_unlabeled - 1)
                    Q_index.append(num)  # 添加选择的样本索引

        # 将选择的索引从局部的`Q_index`转换回原始未标注数据集的索引
        Q_index = [self.U_index[idx] for idx in Q_index]

        return Q_index  # 返回选择的样本索引