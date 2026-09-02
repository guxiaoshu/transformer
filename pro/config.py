
from dataclasses import dataclass
import torch
import os


@dataclass
class EduConfig:

    data_dir: str = "D:/Codefield/Transformer/huggingface/datasets/Helsinki-NLP___opus-100"

    src_lang: str = "zh"#源语言
    tgt_lang: str = "en"#目标语言

    max_train_samples: int = 1000000   # 训练样本数
    min_seq_len: int = 2  # 最小序列长度
    max_seq_len: int = 128  # 最大序列长度

    vocab_size: int = 32000 # 联合词表大小
    tokenizer_path: str = os.path.join(os.path.dirname(__file__), "checkpoints", "spm.model")# 词表保存路径

    batch_size: int = 64#一个batch送进去几句子（8G显存：32→64 喂饱GPU，约提速30%）

    d_model: int = 512 # 隐藏维度（pro：256 → 512）

    n_heads: int = 16# 注意力头数（pro：8 → 16）

    n_encoder_layers: int = 6# 编码器层数（pro：3 → 6）

    n_decoder_layers: int = 6# 解码器层数（pro：3 → 6）
    d_ff: int = 2048 # FFN隐藏层维度（pro：512 → 2048）

    dropout: float = 0.1 # Dropout比例
    max_pos_len: int = 128# 位置编码最大长度（pro：64 → 128）
    verbose: bool = True# 要不要打印的开关


    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    #检测一下有没有cuda，必须有用cpu训练半天训不完

    label_smoothing: float = 0.1 # 标签平滑系数：10%的概率均匀分配给非正确答案
    lr: float = 5e-5 # 初始学习率（pro：模型大了10倍，1e-4 → 5e-5）

    adam_beta1: float = 0.9 # β1: 一阶动量系数（控制"惯性"），0.9 = 标准值

    adam_beta2: float = 0.98 # β2: 二阶动量系数

    adam_eps: float = 1e-9
    # ε: 防止除零的小常数

    weight_decay: float = 0.01 # 权重衰减系数：L2正则化的强度，0.01 = 适度正则化


    warmup_steps: int = 10000# 预热步数（pro：100万条下 2000 → 10000，约1个epoch）

    epochs: int = 30# 训练几轮？
    grad_clip: float = 1.0# 梯度裁剪阈值：限制梯度L2范数不超过1.0，防止梯度爆炸

    log_interval: int = 50# 隔几个batch打印一次日志？

    save_best: bool = True#保存还是不保存

    # 混合精度 AMP（30系显卡用 fp16）
    use_amp: bool = True

    # 早停，连续 N 个 epoch 的 val loss 没创新低就提前结束
    early_stop_patience: int = 5

    save_dir: str = os.path.join(os.path.dirname(__file__), "checkpoints")
    # 保存目录：模型检查点存放位置

config = EduConfig()
