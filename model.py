"""
================================================================================
Transformer 教育版 — 模型架构模块 (Model)
================================================================================

【本模块在Pipeline中的位置】
  token IDs → [model.py] → logits (每个位置的词表概率分布)

【设计原则】
  完全手写，不依赖 huggingface/transformers 或任何封装好的 Transformer 库。
  只使用 torch.nn 的基础组件（Linear, LayerNorm, Dropout, Embedding）。
  每一层都有 verbose 模式，可打印张量形状变化。

【架构总览（Seq2Seq Transformer）】

  输入中文          输入英文(训练时)
     │                   │
     ▼                   ▼
  Embedding          Embedding          ← 共享词表，同一矩阵
     │                   │
     ▼                   ▼
  PositionalEncoding  PositionalEncoding ← sin/cos 固定编码
     │                   │
     ▼                   │
  Encoder ×N           │               ← Self-Attention ×N
  (Self-Attn+FFN)       │
     │                   │
     ▼                   ▼
  enc_out ──────→  Decoder ×N           ← Self-Attn(causal) + Cross-Attn + FFN
                       │
                       ▼
                   Output Head           ← Linear(vocab_size) → softmax → 预测下一个token

【符号约定（贯穿本文件的张量命名）】
  B = batch_size        批次大小
  S = source length     源语言（中文）序列长度
  T = target length     目标语言（英文）序列长度
  D = d_model           模型隐藏维度（被n_heads整除）
  H = n_heads           注意力头数
  V = vocab_size        词表大小
  d_k = D/H             每个注意力头的维度

【Post-Norm vs Pre-Norm（教育版用Post-Norm）】
  Post-Norm（原始论文结构）：
    x = LayerNorm(x + Sublayer(x))      先做子层，再加残差，最后归一化
    └─ 优点：历史原因，大多数经典实现用这个
    └─ 缺点：梯度可能消失，训练不稳定时需要 warmup

  Pre-Norm（生产版用）：
    x = x + Sublayer(LayerNorm(x))      先归一化，再做子层，再加残差
    └─ 优点：训练更稳定，对学习率不那么敏感
    └─ 缺点：理论上表达能力略弱

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. 位置编码 (Positional Encoding)
# ==============================================================================
# Attention 机制本身不关心 token 的位置（"你好"和"好你"的注意力分数相同）。
# 位置编码给每个位置注入一个唯一的"位置信号"，让模型知道 token 的顺序。
#
# 公式（来自 "Attention Is All You Need" 论文第3.5节）：
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# 为什么用 sin/cos 而不用可学习的位置嵌入？
#   1. 确定性：不需要额外参数，减小过拟合风险
#   2. 外推性：理论上可以外推到比训练时更长的序列
#   3. 相对位置：sin(a+b) = sin(a)cos(b) + cos(a)sin(b)，
#      模型可以通过线性变换获取相对位置信息
# ==============================================================================

class PositionalEncoding(nn.Module):
    """
    正弦位置编码 —— 为每个 token 注入唯一的位置信息。

    【数据流】
      输入: x (B, seq_len, d_model)  ← 词嵌入的输出
      处理: x + PE[0:seq_len]         ← 查表，逐元素相加
      输出: (B, seq_len, d_model)     ← 融合了语义+位置信息

    【关键设计】
      - PE 矩阵在 __init__ 中预计算，存入 register_buffer
      - register_buffer：随模型保存/加载，但不参与梯度更新
      - 不同维度的频率不同（从 1/10000^0 到 1/10000^(d_model-2)/d_model）
        低维编码细粒度位置，高维编码粗粒度位置
    """

    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        """
        【参数说明】
          d_model: 词嵌入维度（必须和模型中其他地方的维度一致）
          max_len: 预计算的最大位置数（超过则无法处理）
          dropout: 对"embedding + PE"的结果做dropout，提高鲁棒性
        """
        super().__init__()

        # ── Dropout：随机关闭一部分神经元，防止过拟合 ──
        # 训练时：随机把10%的值置0，其余值放大1/(1-0.1)倍
        # 推理时：自动关闭（model.eval()后Dropout不做任何操作）
        self.dropout = nn.Dropout(p=dropout)

        # ── 预计算位置编码矩阵 ──
        # pe: (max_len, d_model) 每一行是一个位置的编码向量

        pe = torch.zeros(max_len, d_model)  # 先全填0

        # position: (max_len, 1) → [[0],[1],[2],...,[max_len-1]]
        # 这是每个位置的索引（0-based）
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: (d_model/2,)
        # 公式中的 10000^(2i/d_model) = exp(2i/d_model * -log(10000))
        # = exp(i * -log(10000) / (d_model/2))
        # 这与原论文等价，但计算更高效
        # 例如 d_model=256: div_term = [1.000, 0.931, 0.866, ..., 0.000]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *    # 0, 2, 4, ..., d_model-2
            (-math.log(10000.0) / d_model)           # -log(10000)/d_model
        )

        # ── 填充 pe ──
        # 偶数索引（0, 2, 4, ...）= sin(position * div_term)
        # 每个位置乘以不同频率的缩放因子，高频=快速震荡=区分邻近位置
        pe[:, 0::2] = torch.sin(position * div_term)   # (max_len, d_model/2)

        # 奇数索引（1, 3, 5, ...）= cos(position * div_term)
        # 用cos让相邻维度正交（sin和cos有90度相位差）
        pe[:, 1::2] = torch.cos(position * div_term)   # (max_len, d_model/2)

        # 在第0维加 batch 维度 → (1, max_len, d_model)
        # 这样 forward 时可以直接和 (B, seq, d_model) 相加（广播到所有batch）
        pe = pe.unsqueeze(0)

        # register_buffer: 像 parameter 一样随模型保存/移动设备，
        # 但不参与梯度计算（requires_grad=False）
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        【前向传播】
          把"纯语义的embedding"变成"带位置信息的embedding"。

        【为什么是相加而不是拼接】
          相加：x + pe → 维度保持 d_model
          拼接：concat(x, pe) → 维度变成 2*d_model，需要额外的投影层
          实践证明相加效果一样好，且参数更少。

        【参数说明】
          x: (B, seq_len, d_model) 词嵌入的输出
          verbose: 是否打印形状信息（用于学习理解）
        """
        # self.pe[:, :seq_len, :] 取前 seq_len 个位置
        # 广播加法：(B, seq, D) + (1, seq, D) → (B, seq, D)
        seq_len = x.size(1)
        out = x + self.pe[:, :seq_len, :]

        if verbose:
            print(f"  [PosEncoding] 输入: {x.shape} → 加位置编码 → 输出: {out.shape}")

        # dropout 之后返回
        return self.dropout(out)


# ==============================================================================
# 2. 多头注意力 (Multi-Head Attention)
# ==============================================================================
# 这是 Transformer 的核心——让每个 token 都能"看到"并"选择性关注"
# 序列中的其他 token。
#
# 为什么需要"多头"？
#   单头只能学到一种"关注模式"（比如"关注相邻的词"）。
#   多头让模型同时学到多种模式：
#     Head 1: 关注主语-谓语关系
#     Head 2: 关注形容词-名词关系
#     Head 3: 关注介词短语
#     Head 4: ...
#
# 公式：
#   Attention(Q,K,V) = softmax(QK^T / √d_k + mask) · V
#
# 各成分的含义：
#   Q (Query):  "我在找什么？" — 当前token想找的信息
#   K (Key):    "我是什么？"   — 每个token的"标签"
#   V (Value):  "我有什么？"   — 每个token的"内容"
#   QK^T:       匹配分数矩阵，"我有多关注你？"
#   √d_k:       缩放因子，防止点积太大导致 softmax 进入饱和区
#   mask:       -inf 屏蔽某些位置（padding / 未来token）
#   softmax:    归一化为概率分布（每行求和=1）
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    多头缩放点积注意力 — Transformer 的核心组件。

    【数据流（以自注意力为例）】
      输入 x: (B, seq, D=256)
        ├─→ W_q(x) → Q: (B, seq, 256)
        ├─→ W_k(x) → K: (B, seq, 256)
        └─→ W_v(x) → V: (B, seq, 256)
        ↓
      拆头：reshape → (B, H=4, seq, d_k=64)
        ↓
      scores = QK^T / √64 → (B, 4, seq, seq)   每个头独立计算
        ↓
      scores + mask (padding→-inf)  ← 屏蔽无效位置
        ↓
      attn = softmax(scores) → (B, 4, seq, seq)  归一化为权重
        ↓
      context = attn · V → (B, 4, seq, 64)       加权求和
        ↓
      合并头：reshape → (B, seq, 256)
        ↓
      W_o(x) → (B, seq, 256)                     最终输出
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        【参数说明】
          d_model: 模型总维度，必须能被 n_heads 整除
          n_heads: 注意力头数
                   教育版=8（为什么是8？原论文用8，256/8=32维每头，刚好）
          dropout: 对注意力权重做dropout，防止某个token过度关注另一个

        【断言检查】
          如果 d_model 不能被 n_heads 整除，就无法均匀分配到各个头。
          例如 256/8=32 ✓, 256/7=36.57 ✗
        """
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # 每个头的维度，例如 256/8=32

        # ── 四个线性投影矩阵 ──
        # 这些是可学习参数（训练中不断更新）

        # W_q: 把输入投影到 Query 空间
        # 形状 (d_model, d_model)，例如 (256, 256)
        # bias=False: Transformer原论文不用bias，减少参数
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        # W_k: 投影到 Key 空间
        self.W_k = nn.Linear(d_model, d_model, bias=False)

        # W_v: 投影到 Value 空间
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # W_o: 把拼接后的多头输出投影回 d_model 维度
        # 为什么需要这一步？拼接后已经是 d_model 维了，
        # 但这层让模型学习"如何融合各个头的信息"
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # ── Dropout ──
        # 对 softmax 后的注意力权重做Dropout
        # 相当于随机"切断"某些token之间的注意力连接
        self.dropout = nn.Dropout(p=dropout)

        # ── 存储最近一次计算的注意力权重 ──
        # 用于训练后的可视化和分析
        # detach() 后的张量，不参与梯度计算
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        【拆头函数】把 d_model 维度拆成 n_heads × d_k

        【张量变换】
          (B, seq, d_model=256)
            → reshape → (B, seq, 8, 32)   ← 最后一维变成 [头数, 每头维度]
            → transpose(1,2) → (B, 8, seq, 32)  ← 把"头数"维度提前

        为什么需要在第2维？
          因为后续矩阵乘法需要 (..., seq, d_k) 形状。
          把 n_heads 放在 batch 后面，每个头独立做矩阵运算，
          利用 PyTorch 的广播机制，免去手动 for 循环。
        """
        B, seq_len, _ = x.shape
        # reshape: 把 d_model 拆成 n_heads × d_k
        x = x.view(B, seq_len, self.n_heads, self.d_k)
        # transpose: 交换第1维(seq)和第2维(n_heads)
        return x.transpose(1, 2)   # → (B, n_heads, seq_len, d_k)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        【合并头函数】拆头操作的逆过程

        (B, n_heads, seq, d_k)
          → transpose(1,2) → (B, seq, n_heads, d_k)
          → contiguous()   ← 让内存连续（transpose后内存不连续，view需要连续内存）
          → view → (B, seq, d_model)
        """
        B, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)              # → (B, seq_len, n_heads, d_k)
        x = x.contiguous()                 # → 内存连续化
        return x.view(B, seq_len, self.d_model)   # → (B, seq_len, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, verbose: bool = False) -> torch.Tensor:
        """
        【前向传播 — 完整的注意力计算流程】

        【参数说明】
          query: (B, Q_len, D)  查询 — "我在找什么？"
          key:   (B, K_len, D)  键   — "我是什么？"
          value: (B, V_len, D)  值   — "我有什么？"
          mask:  (B,1,1,K_len) 或 (1,1,Q_len,K_len)
                 - 值0的位置 → 有效（不修改attention分数）
                 - 值-inf的位置 → 屏蔽（softmax后权重≈0）
          verbose: 如果True，打印每一步的形状

        【自注意力 vs 交叉注意力】
          自注意力: Q=K=V=同一个序列  → 编码器内部 / 解码器（掩码）内部
          交叉注意力: Q=解码器, K=V=编码器 → 解码器关注源语言

        【返回值】
          (B, Q_len, d_model) — 每个query位置的上下文表示
        """
        B = query.size(0)

        if verbose:
            print(f"  [MultiHeadAttn] Q:{query.shape} K:{key.shape} V:{value.shape}")

        # ── 步骤1：线性投影 → 拆头 ──
        # 三个独立的线性变换，把输入投影到 Q/K/V 空间
        # 然后拆成多头形状
        Q = self._split_heads(self.W_q(query))    # (B, H, Q_len, d_k)
        K = self._split_heads(self.W_k(key))      # (B, H, K_len, d_k)
        V = self._split_heads(self.W_v(value))    # (B, H, V_len, d_k)

        if verbose:
            print(f"    → 拆头后 Q:{Q.shape} K:{K.shape} V:{V.shape}")

        # ── 步骤2：计算注意力分数 ──
        # QK^T: 矩阵乘法，(B,H,Q_len,d_k) × (B,H,d_k,K_len) → (B,H,Q_len,K_len)
        # 含义：第i个query与第j个key的"匹配程度"
        #
        # 除以 √d_k：缩放因子
        #   原因：d_k 越大，QK^T 的方差越大（每个元素是 d_k 个乘积的和）
        #   方差的增大会让 softmax 输出趋于 one-hot(梯度趋近于0)
        #   除以 √d_k 让方差保持 ≈1，稳定训练
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # ── 步骤3：应用mask（在softmax之前！）──
        # mask 加到 scores 上：
        #   - 有效位置 (mask=0)   → 分数不变
        #   - 屏蔽位置 (mask=-inf) → 分数变为 -inf → softmax后=0
        # 为什么是加而不是乘？
        #   因为 softmax 的定义：e^x / Σe^x
        #   如果 x=-inf，则 e^(-inf)=0，达到了"完全屏蔽"的效果
        #   如果用乘法（×0），e^0=1，仍然有非零权重
        if mask is not None:
            scores = scores + mask

        # ── 步骤4：Softmax 归一化 ──
        # 沿最后一维(K_len)做softmax：
        #   每个 query 对所有 key 的注意力权重之和 = 1
        attn_weights = F.softmax(scores, dim=-1)

        # 对注意力权重做 dropout
        # 随机关闭一些注意力连接 → 强迫模型学习多样化的关注模式
        attn_weights = self.dropout(attn_weights)

        # 保存注意力权重（不参与梯度），供后续可视化
        self.attn_weights = attn_weights.detach()

        # ── 步骤5：加权求和 ──
        # attn_weights · V: (B,H,Q_len,K_len) × (B,H,K_len,d_k) → (B,H,Q_len,d_k)
        # 含义：对每个 query，把所有 value 按注意力权重加权求和
        # 结果 context 是"query 从所有 key 处收集到的信息"
        context = torch.matmul(attn_weights, V)

        if verbose:
            print(f"    → scores:{scores.shape} attn:{attn_weights.shape} context:{context.shape}")

        # ── 步骤6：合并头 → 输出投影 ──
        # 把多头拼回 d_model 维度，然后过一个线性层
        output = self.W_o(self._combine_heads(context))

        if verbose:
            print(f"    → 输出: {output.shape}")

        return output


# ==============================================================================
# 3. 前馈神经网络 (Position-wise Feed-Forward Network)
# ==============================================================================
# 注意力层负责"不同位置之间的信息交换"，
# FFN 负责"单个位置内部的信息处理"。
# 两者结合，模型既有全局视野（注意力）又有局部处理能力（FFN）。
#
# 公式：FFN(x) = ReLU(x·W1 + b1)·W2 + b2
#
# 为什么需要 FFN？
#   注意力是线性的（加权求和），FFN引入了非线性(ReLU)，
#   让模型能学习更复杂的特征变换。
#
# d_ff 通常 = 4 × d_model（原论文），教育版用 2× 以减小模型。
# ==============================================================================

class FeedForward(nn.Module):
    """
    位置独立的前馈网络 —— 对每个 token 独立做相同的非线性变换。

    【为什么是"位置独立"（position-wise）】
      nn.Linear 对 (B, seq, D) 中最后维 D 做变换，
      对序列中每个 token 使用完全相同的参数。
      相当于 1×1 卷积（在序列维度上）。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        【参数说明】
          d_model: 输入/输出维度 = 256
          d_ff:    中间隐藏层维度 = 512（2×d_model）
                   原论文用4×(2048/512)，教育版用2×以减小模型
          dropout: 对隐藏层输出做 dropout
        """
        super().__init__()

        # 第一层：d_model → d_ff（扩张）
        # 先升维，在高维空间做非线性变换，再降维回来
        self.linear1 = nn.Linear(d_model, d_ff)

        # 第二层：d_ff → d_model（收缩）
        # 降维回原始维度，保持残差连接可行
        self.linear2 = nn.Linear(d_ff, d_model)

        # ReLU 之后的 dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        【前向传播】
          x (B,seq,256) → linear1 → (B,seq,512) → ReLU → dropout
          → linear2 → (B,seq,256)

        【为什么先升维再降维】
          高维空间提供更大的表示容量，
          模型可以学到更丰富的特征组合。
          可以想象为：
            256维 → 展开成512个特征 → 非线性筛选 → 压缩回256维
        """
        # 升维 + 非线性
        out = self.linear2(
            self.dropout(
                F.relu(
                    self.linear1(x)
                )
            )
        )

        if verbose:
            print(f"  [FFN] {x.shape} → {self.linear1.out_features} → ReLU → {x.shape}")

        return out


# ==============================================================================
# 4. 编码器层 (Encoder Layer)
# ==============================================================================
# 编码器层的结构（Post-Norm）：
#   x = x + Dropout(MultiHeadAttention(LayerNorm(x)))   ← 子层1: 自注意力
#   x = x + Dropout(FeedForward(LayerNorm(x)))           ← 子层2: FFN
#
# 注意：这是 Post-Norm（先子层，后归一化）。教育版用这个。
# ==============================================================================

class EncoderLayer(nn.Module):
    """
    Transformer 编码器层 —— 一层编码器包含一个自注意力 + 一个FFN。

    【为什么需要多个编码器层】
      浅层（第1-2层）：学到浅层特征（词性、短语结构）
      中层（第3-4层）：学到句法特征
      深层（第5-6层）：学到语义特征
      教育版用3层，覆盖基础特征。

    【残差连接 (Residual Connection)】
      x = LayerNorm(x + Sublayer(x))
      为什么要加 x？
        1. 梯度可以直接流过残差连接，避免深层网络的梯度消失
        2. 让模型选择"保留原始信息"还是"使用变换后的信息"
        3. 实践中极其重要：没有残差连接，深层 Transformer 几乎无法训练
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        """
        【参数全部透传给子组件】
          d_model: 输入/输出维度
          n_heads: 注意力头数
          d_ff: FFN隐藏层维度
          dropout: 所有 dropout 层的丢弃率
        """
        super().__init__()

        # 子层1：多头自注意力（Q=K=V=自身）
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 子层2：前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorm ×2（每个子层一个）
        # LayerNorm 对每个样本的每个位置独立归一化：
        #   y = (x - mean) / std * γ + β
        # 好处：稳定训练，加速收敛
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout（用于残差连接后，子层输出上）
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None,
                verbose: bool = False) -> torch.Tensor:
        """
        【前向传播】

        【参数说明】
          x: (B, S, D)  输入序列（来自上一层编码器或词嵌入）
          src_mask: (B, 1, 1, S)  源语言padding mask
                    0=有效, -inf=屏蔽
          verbose: 是否打印形状

        【返回】
          (B, S, D)  丰富了语义信息的序列表示
        """
        if verbose:
            print(f"  [EncoderLayer] 输入: {x.shape}")

        # ── 子层1：自注意力 + 残差 + LayerNorm ──
        # self.self_attn(x, x, x): Q=K=V=x → 自己注意自己
        attn_out = self.self_attn(x, x, x, mask=src_mask, verbose=verbose)
        # 残差连接：x + dropout(attn_out)
        # 然后 LayerNorm 归一化
        x = self.norm1(x + self.dropout(attn_out))

        # ── 子层2：FFN + 残差 + LayerNorm ──
        ffn_out = self.ffn(x, verbose=verbose)
        x = self.norm2(x + self.dropout(ffn_out))

        if verbose:
            print(f"  [EncoderLayer] 输出: {x.shape}")

        return x


# ==============================================================================
# 5. 解码器层 (Decoder Layer)
# ==============================================================================
# 解码器比编码器多一个"交叉注意力"子层，用来关注编码器的输出。
# 同时自注意力是"掩码"的（不能看未来token）。
#
# 结构（Post-Norm）：
#   x = x + Dropout(MaskedSelfAttention(LayerNorm(x)))     ← 子层1: 掩码自注意力
#   x = x + Dropout(CrossAttention(LayerNorm(x), enc_out))  ← 子层2: 交叉注意力
#   x = x + Dropout(FeedForward(LayerNorm(x)))              ← 子层3: FFN
# ==============================================================================

class DecoderLayer(nn.Module):
    """
    Transformer 解码器层 —— 三层结构：掩码自注意 + 交叉注意 + FFN。

    【三个子层的分工】
      子层1 (Masked Self-Attn): "我已经生成了哪些词？它们之间有什么关系？"
      子层2 (Cross-Attn):        "源语言说了什么？我应该把注意力放在哪里？"
      子层3 (FFN):               "基于以上信息，我应该输出什么特征？"
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 子层1：掩码自注意力（不能偷看未来token）
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 子层2：交叉注意力（Q=解码器, K/V=编码器输出）
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 子层3：前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # LayerNorm ×3
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None,
                verbose: bool = False) -> torch.Tensor:
        """
        【前向传播】

        【参数说明】
          x: (B, T, D)        解码器当前输入（训练时=shifted targets，推理时=历史生成token）
          enc_out: (B, S, D)  编码器输出（整个源语言序列的编码）
          tgt_mask: (B,1,T,T) 因果mask（下三角=0, 上三角=-inf）
          src_mask: (B,1,1,S) 源语言padding mask
          verbose: 是否打印形状

        【tgt_mask 为什么是因果的】
          训练时解码器一次性看到整个目标序列（Teacher Forcing），
          但必须防止第i个位置"偷看"第i+1个位置的答案。
          因果 mask 确保第i个位置只能看到位置1..i。

        【返回】
          (B, T, D)  每个位置的下一个token特征
        """
        if verbose:
            print(f"  [DecoderLayer] 输入: {x.shape}, enc_out: {enc_out.shape}")

        # ── 子层1：掩码自注意力 ──
        # Q=K=V=x，但加因果mask防止看到未来token
        attn_out = self.self_attn(x, x, x, mask=tgt_mask, verbose=verbose)
        x = self.norm1(x + self.dropout(attn_out))

        # ── 子层2：交叉注意力 ──
        # Q=解码器当前状态, K=V=编码器输出
        # 让解码器"查阅"源语言信息
        cross_out = self.cross_attn(x, enc_out, enc_out, mask=src_mask, verbose=verbose)
        x = self.norm2(x + self.dropout(cross_out))

        # ── 子层3：FFN ──
        ffn_out = self.ffn(x, verbose=verbose)
        x = self.norm3(x + self.dropout(ffn_out))

        if verbose:
            print(f"  [DecoderLayer] 输出: {x.shape}")

        return x


# ==============================================================================
# 6. 完整 Seq2Seq Transformer
# ==============================================================================
# 把 Embedding、位置编码、N×Encoder、N×Decoder、输出头 组装成完整模型。
# ==============================================================================

class Seq2SeqTransformer(nn.Module):
    """
    序列到序列 Transformer 翻译模型。

    【完整数据流（训练模式）】
      src IDs (B,S)                tgt IDs (B,T)
          │                              │
          ▼                              ▼
      Embedding · √D               Embedding · √D
          │                              │
          ▼                              ▼
      PositionalEncoding           PositionalEncoding
          │                              │
          ▼                              │
      Encoder ×N                        │
      (Self-Attn)                        │
          │                              │
          ▼                              ▼
      enc_out ──────────→  Decoder ×N
                              │
                              ▼
                          Linear(V) → logits (B,T,V)

    【关键设计决策】
      1. 共享词表：源语言和目标语言用同一个 Embedding 矩阵
         好处：参数共享，两种语言的语义空间对齐
      2. 缩放嵌入：embed × √d_model
         原因：embedding 初始化时方差很小（~N(0,1)），
               PE 的值在 [-1, 1] 范围，两者量级不匹配。
               缩放 embedding 让两者在相同量级上。
      3. Xavier初始化：权重初始化为均匀分布
         好处：保持前向传播和反向传播的方差一致，训练更稳定
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_encoder_layers: int, n_decoder_layers: int, d_ff: int,
                 dropout: float = 0.1, max_len: int = 64, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        # ── 共享词嵌入矩阵 ──
        # padding_idx=0: PAD token 的嵌入始终为0，梯度始终为0
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 缩放因子：让 embedding 和 positional encoding 在相似量级
        # 论文原始做法，实践证明有小幅提升
        self.embed_scale = math.sqrt(d_model)   # 例如 √256 = 16

        # ── 位置编码 ──
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # ── 编码器（N层） ──
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # ── 解码器（N层） ──
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # ── 输出投影头 ──
        # 把 d_model 维的特征映射到 vocab_size 维的 logits
        # 不直接用 embedding.weight 的转置（weight tying），
        # 因为这里可能 vocab_size ≠ d_model，用独立层更灵活
        self.output_proj = nn.Linear(d_model, vocab_size)

        # ── 参数初始化 ──
        self._init_parameters()

        # 打印模型信息
        try:
            from config import config
            if config.verbose:
                print(f"\n[Seq2SeqTransformer] 模型创建完毕:")
                print(f"  词表大小:      {vocab_size}")
                print(f"  隐藏维度:      {d_model}")
                print(f"  注意力头数:    {n_heads}")
                print(f"  编码器层数:    {n_encoder_layers}")
                print(f"  解码器层数:    {n_decoder_layers}")
                print(f"  FFN 维度:      {d_ff}")
                print(f"  总参数量:      {sum(p.numel() for p in self.parameters()):,}")
        except ImportError:
            pass

    def _init_parameters(self):
        """
        【Xavier/Glorot 参数初始化】

        为什么要手动初始化？
          PyTorch 默认的初始化方式不一定是 Transformer 的最优选择。
          Xavier 初始化保持前向和反向传播的方差稳定，避免梯度爆炸/消失。

        初始化策略：
          - 多维参数(dim>1): Xavier均匀分布
            适用于 nn.Linear 的权重矩阵
          - 一维参数(bias): 保持默认零初始化
            nn.Linear 的 bias 和 nn.LayerNorm 的 weight/bias
        """
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_ 在原位修改参数值
                # 均匀分布 U[-a, a]，其中 a = √(6/(fan_in + fan_out))
                nn.init.xavier_uniform_(p)

    # ==========================================================================
    # 6a. Mask 生成函数
    # ==========================================================================

    @staticmethod
    def create_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        """
        【创建 Padding Mask】
        把 (B, S) 的布尔mask转为 (B, 1, 1, S) 的注意力mask。

        【转换规则】
          True (有效位置)  → 0      → 不影响attention分数
          False (PAD位置)  → -inf   → softmax后权重=0

        【维度变换】
          (B, S) → unsqueeze(1) → (B, 1, S) → unsqueeze(2) → (B, 1, 1, S)
          这4个维度分别对应 [batch, heads, query_len, key_len]
          广播到所有head和所有query位置。

        【.float().log() 的技巧】
          布尔mask → float (True=1.0, False=0.0)
          log(1.0) = 0       → 有效
          log(0.0) = -inf    → 屏蔽
          这种方式比手动填充 -inf 更简洁优雅。
        """
        return mask.unsqueeze(1).unsqueeze(2).float().log()

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        【创建因果 Mask（下三角矩阵）】
        确保解码器第 i 个位置只能看到位置 1..i，不能偷看未来。

        【矩阵形式（seq_len=5）】
              0  1  2  3  4  (key位置)
          0 [ 0 -∞ -∞ -∞ -∞]   ← 位置0只能看自己
          1 [ 0  0 -∞ -∞ -∞]   ← 位置1能看0和1
          2 [ 0  0  0 -∞ -∞]   ← 位置2能看0,1,2
          3 [ 0  0  0  0 -∞]
          4 [ 0  0  0  0  0]

        【实现方式】
          torch.triu(矩阵, diagonal=1): 保留对角线上方的元素
          先创建全 -inf 矩阵，再用 triu 让上三角保持 -inf
          下三角和对角线初始化为别的值 → 0（torch.ones）
        """
        # torch.triu: 上三角矩阵
        #   diagonal=1: 从对角线上面第一条开始保留
        #   结果：下三角=0, 上三角=-inf
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        # unsqueeze: 加 batch 和 head 维度 → (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_tgt_mask(tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        【创建解码器完整 Mask = 因果 mask + padding mask】

        为什么需要两个 mask 叠加？
          - 因果 mask: 防止看到未来token（所有batch共享同一因果mask）
          - Padding mask: 防止关注到padding位置（每个样本可能不同）

        两个mask相加后：被屏蔽的位置 = -inf + (-inf) = -inf（仍然是屏蔽）
        有效位置 = 0 + 0 = 0（不修改attention分数）
        """
        B, T = tgt_padding_mask.shape
        device = tgt_padding_mask.device

        # 因果mask
        causal = Seq2SeqTransformer.create_causal_mask(T, device)  # (1, 1, T, T)

        # Padding mask: (B, 1, 1, T)
        pad = tgt_padding_mask.unsqueeze(1).unsqueeze(2).float().log()

        # 广播加法: (1,1,T,T) + (B,1,1,T) → (B,1,T,T)
        return causal + pad

    # ==========================================================================
    # 6b. 前向传播（训练模式）
    # ==========================================================================

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                verbose: bool = False) -> torch.Tensor:
        """
        【训练模式前向传播 — Teacher Forcing】

        【数据流简述】
          src(B,S) → embed→scale→posenc → ×N EncoderLayer → enc_out(B,S,D)
          tgt(B,T) → embed→scale→posenc → ×N DecoderLayer → dec_out(B,T,D)
          dec_out → Linear(V) → logits(B,T,V)

        【Teacher Forcing 详解】
          tgt_input 是目标的"前N-1个token"（去掉最后的EOS）
          模型对每个位置i预测第i+1个token
          因为用了因果mask，位置i只能看到≤i的输入，不能偷看答案

        【参数说明】
          src: (B, S)  源语言token IDs
          tgt_input: (B, T)  目标语言输入token IDs
          src_mask: (B, S)  源语言有效mask (True=有效)
          tgt_mask: (B, T)  目标语言有效mask
          verbose: 打印每步形状

        【返回】
          logits: (B, T, V)  每个位置对词表中每个token的"原始分数"
                  需要外部做 softmax/argmax 才能得到预测
        """
        if verbose:
            print("=" * 60)
            print("  模型前向传播 (Verbose Mode)")
            print("=" * 60)
            print(f"  src: {src.shape} (token IDs)")
            print(f"  tgt_input: {tgt_input.shape} (token IDs)")

        # ── 步骤1：创建 Attention Mask ──
        # 把布尔mask转为带 -inf 的浮点mask
        src_attn_mask = self.create_padding_mask(src_mask) \
            if src_mask is not None else None
        tgt_attn_mask = self.create_tgt_mask(tgt_mask) \
            if tgt_mask is not None else None

        if verbose:
            print(f"\n  src_attn_mask: "
                  f"{src_attn_mask.shape if src_attn_mask is not None else 'None'}")
            print(f"  tgt_attn_mask: "
                  f"{tgt_attn_mask.shape if tgt_attn_mask is not None else 'None'}")

        # ── 步骤2：词嵌入 + 缩放 ──
        # 为什么不直接用 embedding？
        # 因为 embedding 初始值很小（~0均值, ~1标准差），
        # 而 positional encoding 在 [-1,1] 范围。
        # 乘以 √d_model 放大了embedding，让两者在相同量级。
        src_emb = self.embedding(src) * self.embed_scale  # (B, S, D)
        tgt_emb = self.embedding(tgt_input) * self.embed_scale  # (B, T, D)

        if verbose:
            print(f"\n  src_emb (嵌入+缩放): {src_emb.shape}")
            print(f"  tgt_emb (嵌入+缩放): {tgt_emb.shape}")

        # ── 步骤3：编码器 ──
        # x 经过位置编码，然后依次通过N个EncoderLayer
        x = src_emb
        x = self.pos_encoding(x)   # 注入位置信息
        for i, layer in enumerate(self.encoder_layers):
            if verbose:
                print(f"\n[Encoder Layer {i+1}/{len(self.encoder_layers)}]")
            x = layer(x, src_mask=src_attn_mask, verbose=verbose)
        enc_out = x   # (B, S, D)

        if verbose:
            print(f"\n[Encoder] 最终输出: {enc_out.shape}")

        # ── 步骤4：解码器 ──
        x = tgt_emb
        x = self.pos_encoding(x)
        for i, layer in enumerate(self.decoder_layers):
            if verbose:
                print(f"\n[Decoder Layer {i+1}/{len(self.decoder_layers)}]")
            x = layer(x, enc_out,
                      tgt_mask=tgt_attn_mask, src_mask=src_attn_mask,
                      verbose=verbose)
        dec_out = x

        if verbose:
            print(f"\n[Decoder] 最终输出: {dec_out.shape}")

        # ── 步骤5：输出投影 ──
        # 把 d_model 维 → vocab_size 维
        # 每个位置得到一个词表大小的向量，每个值=该token的"分数"
        # 分数越高=模型越倾向于选择这个token
        logits = self.output_proj(dec_out)  # (B, T, V)

        if verbose:
            print(f"\n  logits (输出): {logits.shape}")
            print("=" * 60)

        return logits

    # ==========================================================================
    # 6c. 推理方法（自回归解码）
    # ==========================================================================

    def encode_for_inference(self, src: torch.Tensor, src_mask: torch.Tensor,
                             verbose: bool = False) -> torch.Tensor:
        """
        【推理用编码器】
        和训练时编码器的区别：不重复计算（推理时只跑一次编码器）。
        训练时每个句子对跑一次编码器，推理时也跑一次，
        但推理时解码器要循环多次。

        【参数说明】
          src: (1, S)  单个句子
          src_mask: (1, S)

        【返回】
          enc_out: (1, S, D)  编码器输出（缓存用于解码器每次查询）
        """
        src_emb = self.embedding(src) * self.embed_scale
        src_emb = self.pos_encoding(src_emb)
        src_attn_mask = self.create_padding_mask(src_mask)

        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask=src_attn_mask)

        return src_emb   # enc_out

    def decode_step(self, tgt_token: torch.Tensor, enc_out: torch.Tensor,
                    tgt_mask: torch.Tensor, src_mask: torch.Tensor,
                    past_len: int, verbose: bool = False) -> torch.Tensor:
        """
        【推理用单步解码器】
        自回归生成：每次调用生成一个token。

        【为什么逐步生成】
          训练时用Teacher Forcing，整个目标序列一次性输入。
          推理时没有"正确答案"可用，只能自己生成了上一步的token,
          再把它追加到输入序列末尾，预测下一个token。

        【注意：效率问题】
          每次调用都要重算整个序列（包括历史token），复杂度 O(T²)。
          生产系统会用 KV Cache 优化，缓存已计算的 K/V，避免重算。
          教育版为了代码清晰省略了这个优化。

        【参数说明】
          tgt_token: (1, past_len+1)  历史已生成token + 最新token
          enc_out: (1, S, D)  编码器输出
          tgt_mask: (1, past_len+1)  目标序列mask
          src_mask: (1, S)  源mask（注意维度要和编码器一致）
          past_len: 已生成的token数（仅用于verbose，可删除）
          verbose: 是否打印

        【返回】
          (1, 1, D)  最后一步的解码器输出（只取最后一个位置）
        """
        # 嵌入 + 位置编码
        tgt_emb = self.embedding(tgt_token) * self.embed_scale
        tgt_emb = self.pos_encoding(tgt_emb)

        # 创建mask
        tgt_attn_mask = self.create_tgt_mask(tgt_mask)
        # src_mask 扩展batch维度以匹配
        src_attn_mask = self.create_padding_mask(src_mask.expand(1, -1))

        # 通过解码器
        for layer in self.decoder_layers:
            tgt_emb = layer(tgt_emb, enc_out,
                            tgt_mask=tgt_attn_mask,
                            src_mask=src_attn_mask,
                            verbose=verbose)

        # 只取最后一步的输出 → (1, 1, D)
        return tgt_emb[:, -1:, :]

    # ==========================================================================
    # 6d. 获取注意力权重（可视化用）
    # ==========================================================================

    def get_attention_weights(self):
        """
        【获取所有解码器层的注意力权重】
        用于 --weight 模式下可视化注意力热力图。

        【返回格式】
          [
            {                           ← 第0层解码器
              "self":  Tensor (B,H,T,T),  ← 自注意力权重
              "cross": Tensor (B,H,T,S),  ← 交叉注意力权重
            },
            {                           ← 第1层解码器
              ...
            },
            ...
          ]

        【注意】
          self.attn_weights 是在最近一次 forward 中存储的，
          调用 get_attention_weights 之前需要先跑一次 decode_step。
        """
        weights = []
        for layer in self.decoder_layers:
            weights.append({
                "self": layer.self_attn.attn_weights,
                "cross": layer.cross_attn.attn_weights,
            })
        return weights
