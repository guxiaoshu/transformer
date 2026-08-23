

from dataclasses import dataclass
import torch
import os


@dataclass
class EduConfig:

    data_dir: str = "D:/Codefield/Transformer/huggingface/datasets/Helsinki-NLP___opus-100"

    src_lang: str = "zh"#源语言
    tgt_lang: str = "en"#目标语言

    max_train_samples: int = 50000   # 训练样本数
    min_seq_len: int = 2  # 最小序列长度
    max_seq_len: int = 64  # 最大序列长度

    vocab_size: int = 10000 # 联合词表大小

    tokenizer_path: str = os.path.join(os.path.dirname(__file__), "checkpoints", "tokenizer.json")# 词表保存路径

    batch_size: int = 64#一个batch送进去几句子

    d_model: int = 256 # 隐藏维度

    n_heads: int = 8# 注意力头数

    n_encoder_layers: int = 3# 编码器层数

    n_decoder_layers: int = 3# 解码器层数
    d_ff: int = 512 # FFN隐藏层维度

    dropout: float = 0.1 # Dropout比例
    max_pos_len: int = 64# 位置编码最大长度
    verbose: bool = True# 要不要打印的开关


    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #检测一下有没有cuda，必须有用cpu训练半天训不完

    label_smoothing: float = 0.1 # 标签平滑系数：10%的概率均匀分配给非正确答案
    lr: float = 1e-4 # 初始学习率，直接给经验值

    adam_beta1: float = 0.9 # β1: 一阶动量系数（控制"惯性"），0.9 = 标准值

    adam_beta2: float = 0.98 # β2: 二阶动量系数

    adam_eps: float = 1e-9
    # ε: 防止除零的小常数

    weight_decay: float = 0.01 # 权重衰减系数：L2正则化的强度，0.01 = 适度正则化


    warmup_steps: int = 2000# 预热步数：前2000步线性增长学习率 大概是 3个epoch 

    epochs: int = 30# 训练几轮？
    grad_clip: float = 1.0# 梯度裁剪阈值：限制梯度L2范数不超过1.0，防止梯度爆炸

    log_interval: int = 50# 隔几个batch打印一次日志？

    save_best: bool = True#保存还是不保存

    save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    # 保存目录：模型检查点存放位置

config = EduConfig()
