import os.path as osp
from random import sample
import time
import json
import math
import os
import pandas as pd
import torch

import torch.nn as nn

import torchvision.transforms as T

import time
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.datasets import build_dataset
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
from PIL import Image
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .active_learning.pcb import PCB
from .active_learning.badge import BADGE
from .active_learning.coreset import Coreset
from .active_learning.entropy import Entropy
from .active_learning.learnability import Learnability
from .active_learning.Hkmeans import Hkmeans
_tokenizer = _Tokenizer()



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype


    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if not ctx_init.endswith(".json"):
        prompt_prefix = " ".join(["X"] * n_ctx)

        classnames = [name.replace("_", " ") for name in classnames]
        n_desc_per_cls = None
        if cfg.TRAINER.COOPAL.ASPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.ASPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())

            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")

        elif cfg.TRAINER.COOPAL.AEPATH:
            with open(f"descriptors/descriptors_{cfg.TRAINER.COOPAL.AEPATH}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k,v in desc_dict.items())

            name_lens, prompts = [], []
            for name in classnames:
                name = name.lower()
                for desc in desc_dict[name]:
                    name_lens.append(len(_tokenizer.encode(f"{name}, which is/has {desc}")))
                    prompts.append(prompt_prefix + " " + f"{name}, which is/has {desc}.")

        else:
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        print(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = embedding.size(0)
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, desc_file=None):
        super().__init__()

        # 初始化 PromptLearner 和 CLIP 的组件
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)  # 提示词学习器
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts  # 预先生成的提示词
        self.image_encoder = clip_model.visual  # 图像编码器，使用 CLIP 模型的视觉部分
        self.text_encoder = TextEncoder(clip_model)  # 文本编码器，使用 CLIP 模型的文本部分

        self.logit_scale = clip_model.logit_scale  # CLIP 模型的缩放因子
        self.dtype = clip_model.dtype  # 数据类型（浮动精度）
        self.n_class_desc = []  # 每个类别的描述符数量
        self.n_cls = len(classnames)  # 类别数量
        self.cfg = cfg  # 配置参数

        if desc_file is not None:
            # 如果提供了描述符文件（desc_file），则加载文件中的类别描述符
            with open(f"descriptors/descriptors_{desc_file}", "r") as f:
                desc_dict = json.load(f)
                desc_dict = dict((k.lower(), v) for k, v in desc_dict.items())  # 将类别名称转为小写
            classnames = [name.replace("_", " ") for name in classnames]  # 替换类别名称中的下划线
            for name in classnames:
                name = name.lower()  # 类别名称转为小写
                self.n_class_desc.append(len(desc_dict[name]))  # 将每个类别的描述符数量保存

    def forward(self, image, get_feature=False):
        # 图像特征提取
        image_features = self.image_encoder(image.type(self.dtype))
        # 获取学习到的提示词
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts  # 获取经过标记化的提示词
        # 文本特征提取
        text_features = self.text_encoder(prompts, tokenized_prompts)
        # 如果配置中有 AEPATH，则进行额外处理
        if self.cfg.TRAINER.COOPAL.AEPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                # 将每个类别的文本特征取平均
                tmp.append(text_features[start:start + n].mean(dim=0))
                start += n
            text_features = torch.stack(tmp)
        # 归一化图像和文本特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 图像特征归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 文本特征归一化

        logit_scale = self.logit_scale.exp()  # 对数缩放因子
        logits = logit_scale * image_features @ text_features.t()  # 计算图像与文本的相似度（logits）
        # 如果配置中有 ASPATH，则进行类别 logits 的聚合
        if self.cfg.TRAINER.COOPAL.ASPATH:
            tmp = []
            start = 0
            for n in self.n_class_desc:
                # 对每个类别的 logits 取平均
                tmp.append(torch.sum(logits[:, start:start + n], dim=1) / n)
                start += n
            logits = torch.stack(tmp, dim=1)
        # 如果需要获取特征，则返回 logits 和图像特征
        if get_feature:
            return logits, image_features
        else:
            return logits


@TRAINER_REGISTRY.register()
class ALVLM(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.acc = []

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"加载CLIP中 CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        if cfg.TRAINER.COOPAL.ASPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.ASPATH)
        elif cfg.TRAINER.COOPAL.AEPATH:
            self.model = CustomCLIP(cfg, classnames, clip_model, desc_file=cfg.TRAINER.COOPAL.AEPATH)
        else:
            self.model = CustomCLIP(cfg, classnames, clip_model)



        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(f"prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:

            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    def before_train(self):
        print("INITIALIZE the prompts weights")
        self.build_model()

    def after_train(self):
        print("训练完成")
        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("部署最好的一轮评价模型")
                self.load_model(self.output_dir)
            else:
                print("部署最后一轮模型")
            self.acc.append(self.test())

        # Close writer
        self.close_writer()

    def train(self):
        """通用的训练循环"""
        dataset = build_dataset(self.cfg)
        print(f"数据集大小: {len(dataset.train_x)}")

        unlabeled_dst = dataset.train_x
        U_index = list(range(len(unlabeled_dst)))

        if self.cfg.TRAINER.COOP.CSC:
            n_query = dataset.get_num_classes(unlabeled_dst)
        else:
            n_query = dataset.get_num_classes(unlabeled_dst)
        print("数据集种类数：", n_query)

        select_ratio = 0.2
        n_cand = int(len(unlabeled_dst) * select_ratio)
        print("选择样本的百分比：", select_ratio)

        dataset._train_x = []

        all_rounds_data = []

        # 记录伪标签样本信息
        pseudo_labeled_samples = []

        for i in range(8):  # 可改回8
            start = time.time()

            round_data = {
                "round": i + 1,
                "entropies": [],
                "labels": [],
                "is_pseudo": []  # 新增记录是否为伪标签样本
            }
            from random import sample
            if self.cfg.TRAINER.COOPAL.METHOD == "random" or i == 0:
                idx = sample(U_index, n_query)
                entropy_map = {k: -1 for k in idx}
                is_pseudo_map = {k: False for k in idx}  # 随机选择的不是伪标签
            else:
                method = self.cfg.TRAINER.COOPAL.METHOD
                if method == "entropy":
                    selector = Entropy(self.cfg, self.model, unlabeled_dst, U_index,
                                       dataset.get_num_classes(unlabeled_dst), self.device)
                elif method == "badge":
                    selector = BADGE(self.cfg, self.model, unlabeled_dst, U_index,
                                     dataset.get_num_classes(unlabeled_dst), self.device)
                elif method == "coreset":
                    val_x = dataset._train_x.copy()
                    selector = Coreset(self.cfg, self.model, unlabeled_dst, U_index, val_x,
                                       dataset.get_num_classes(unlabeled_dst))
                elif method == "learnability":
                    selector = Learnability(self.cfg, self.model, unlabeled_dst, U_index,
                                            dataset.get_num_classes(unlabeled_dst), self.device)
                elif method == "hkmeans":
                    selector = Hkmeans(self.cfg, self.model, unlabeled_dst, U_index,
                                       dataset.get_num_classes(unlabeled_dst), self.device)
                else:
                    raise ValueError(f"未实现的方法: {method}")

                query_num = n_query if i == 0 else n_cand
                idx = selector.select(query_num)

                # 获取熵值映射（仅 entropy 和 learnability 有）
                if hasattr(selector, "rank_uncertainty"):
                    uncertainties = selector.rank_uncertainty()
                    if isinstance(uncertainties, (list, np.ndarray)) and len(uncertainties) == len(U_index):
                        entropy_map = dict(zip(U_index, uncertainties))
                    else:
                        print(f"[警告] 熵值长度不等于未标记样本数量：{len(uncertainties)} vs {len(U_index)}")
                        entropy_map = {k: -1 for k in idx}
                else:
                    entropy_map = {k: -1 for k in idx}

                # 获取伪标签信息（仅 learnability 有）
                if hasattr(selector, "_pseudo_selection") and selector._pseudo_selection:
                    is_pseudo_indices = set(selector._pseudo_selection["indices"])
                    is_pseudo_map = {k: (k in is_pseudo_indices) for k in idx}

                    # 记录伪标签样本信息
                    for k in idx:
                        if k in is_pseudo_indices:
                            pseudo_idx = selector._pseudo_selection["indices"].index(k)
                            pseudo_label = selector._pseudo_selection["labels"][pseudo_idx]
                            confidence = selector._pseudo_selection["probs"][pseudo_idx]

                            # 记录伪标签样本信息，不修改原始对象
                            pseudo_labeled_samples.append({
                                "round": i + 1,
                                "sample_idx": k,
                                "global_idx": selector.U_index[k],
                                "pseudo_label": pseudo_label,
                                "confidence": confidence
                            })
                else:
                    is_pseudo_map = {k: False for k in idx}

            # 创建训练集样本的副本，以便修改伪标签
            selected_samples = []
            for k in idx:
                sample = unlabeled_dst[k]

                # 检查是否为伪标签样本
                is_pseudo = is_pseudo_map.get(k, False)

                # 获取原始标签
                if hasattr(sample, 'label'):
                    if isinstance(sample.label, torch.Tensor):
                        original_label = sample.label.item()
                    else:
                        original_label = sample.label
                else:
                    original_label = -1

                # 记录信息到round_data
                round_data["entropies"].append(entropy_map.get(k, -1))
                round_data["labels"].append(original_label)
                round_data["is_pseudo"].append(is_pseudo)

                # 对于伪标签样本，创建新的样本对象
                if is_pseudo:
                    # 找到对应的伪标签信息
                    pseudo_info = next((info for info in pseudo_labeled_samples
                                        if info["sample_idx"] == k and info["round"] == round_data["round"]), None)

                    if pseudo_info:
                        try:
                            # 尝试使用深拷贝
                            import copy
                            new_sample = copy.deepcopy(sample)
                        except:
                            new_sample = sample  # 可能需要更复杂的复制逻辑

                        # 在数据加载器中我们将使用这些伪标签信息
                        new_sample._pseudo_label = pseudo_info["pseudo_label"]
                        selected_samples.append(new_sample)
                    else:
                        selected_samples.append(sample)
                else:
                    selected_samples.append(sample)

                # 从未标记集中移除该样本索引
                U_index.remove(k)

            # 将这批样本添加到训练集
            dataset._train_x.extend(selected_samples)
            all_rounds_data.append(round_data)

            # 输出本轮中伪标签样本的数量
            pseudo_count = sum(round_data["is_pseudo"])
            print(f"第{i + 1}轮 - 伪标签样本数: {pseudo_count}, 总样本数: {len(round_data['is_pseudo'])}")

            # 打印本轮熵值统计
            valid_entropies = [e for e in round_data["entropies"] if e >= 0]
            if valid_entropies:
                avg_entropy = sum(valid_entropies) / len(valid_entropies)
                print(f"第{i + 1}轮 - 平均熵值: {avg_entropy:.4f}")

            assert len(U_index) + len(dataset.train_x) == len(unlabeled_dst)

            # 自定义collate_fn函数来处理伪标签
            def collate_with_pseudo_labels(batch):
                # 检查是否有伪标签
                processed_batch = []
                for sample in batch:
                    if hasattr(sample, '_pseudo_label'):
                        # 创建一个新的样本，用伪标签替换真实标签
                        if isinstance(sample, dict):
                            new_sample = sample.copy()
                            if isinstance(sample.get('label'), torch.Tensor):
                                new_sample['label'] = torch.tensor(sample._pseudo_label)
                            else:
                                new_sample['label'] = sample._pseudo_label
                            processed_batch.append(new_sample)
                        else:
                            # 如果不是字典，需要根据你的数据结构相应地调整
                            # 这里我们假设sample有一个label字段可以被替换
                            try:
                                processed_sample = sample
                                if hasattr(processed_sample, 'label'):
                                    # 保存原始标签到临时属性
                                    processed_sample._original_label = processed_sample.label
                                    # 设置伪标签（注意：这可能会失败，取决于label属性是否可写）
                                    try:
                                        if isinstance(processed_sample.label, torch.Tensor):
                                            processed_sample.label = torch.tensor(sample._pseudo_label)
                                        else:
                                            processed_sample.label = sample._pseudo_label
                                    except:
                                        # 如果无法修改label，使用原始样本
                                        pass
                                processed_batch.append(processed_sample)
                            except:
                                # 如果出现任何错误，使用原始样本
                                processed_batch.append(sample)
                    else:
                        processed_batch.append(sample)

                # 使用默认collate函数处理处理后的批次
                try:
                    from torch.utils.data._utils.collate import default_collate
                    return default_collate(processed_batch)
                except TypeError as e:
                    # 如果默认collate函数失败，尝试一个更简单的方法
                    print(f"警告：默认整理函数失败，回退到简单整理方法。错误：{e}")
                    try:
                        # 尝试将批次简单地作为列表返回
                        return {
                            'img': torch.stack([item['img'] for item in processed_batch]),
                            'label': torch.tensor([item['label'] for item in processed_batch]),
                            # 可能需要添加其他字段...
                        }
                    except:
                        # 如果所有尝试都失败，返回原始批次
                        return batch

            # 创建训练加载器，使用自定义的collate函数
            self.train_loader_x = build_data_loader(
                self.cfg,
                sampler_type=self.cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=True),
                is_train=True,
            )

            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                self.after_epoch()
            self.after_train()
            print(f"第{i + 1}轮训练时间: {time.time() - start:.2f}秒")

        # 统计伪标签总数
        total_pseudo = sum(len([j for j in round_data["is_pseudo"] if j]) for round_data in all_rounds_data)
        print(f"\n=== 总共使用了 {total_pseudo} 个伪标签样本 ===")

        # === 归一化熵统计 ===
        print("\n=== 归一化熵分布统计 ===")
        all_entropies = []
        for round_data in all_rounds_data:
            all_entropies.extend(round_data["entropies"])

        valid_entropies = [e for e in all_entropies if e >= 0]
        if valid_entropies:
            log_c = math.log(self.num_classes)
            normalized_entropies = [e / log_c for e in valid_entropies]

            buckets = {
                "0.0 - 0.4": 0,
                "0.4 - 0.6": 0,
                "0.6 - 1.0": 0
            }

            for ne in normalized_entropies:
                if ne <= 0.4:
                    buckets["0.0 - 0.4"] += 1
                elif ne <= 0.6:
                    buckets["0.4 - 0.6"] += 1
                else:
                    buckets["0.6 - 1.0"] += 1

            total_valid = len(normalized_entropies)
            for k, v in buckets.items():
                ratio = v / total_valid
                print(f"{k}: {v} 个样本，占比 {ratio:.2%}")

        print("=== 准确率结果概览 ===")
        for i in range(len(self.acc)):
            print(f"准确率{i}: {self.acc[i]}")
        print("=======================")