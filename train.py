"""
================================================================================
Transformer 教育版 — 核心训练循环模块 (Train)
================================================================================

【本模块在Pipeline中的位置】
  DataLoader → [train.py] → best_model.pt

【训练策略】
  教师强制 (Teacher Forcing): 解码器每一步输入真实的上一token（而非模型生成的）
  - 训练时: 看到 "I love" 预测 "machine"  (输入真实标签)
  - 推理时: 看到 "I love" 预测 "machine"  (输入自己生成的)
  - 好处: 训练更快、更稳定；坏处: 训练和推理有分布差异 (exposure bias)

【优化器与调度】
  AdamW: Adam的改进版，把权重衰减和梯度更新解耦，效果更好
  Warmup + Cosine Decay: 前N步线性增LR，之后余弦衰减到0
  - 预热: 避免训练初期梯度大导致模型"走偏"
  - 余弦衰减: 平滑降低LR，最后阶段精细调参

【损失函数】
  Label Smoothing Cross-Entropy:
  - 普通CE: model必须100%确定正确答案 → 容易过拟合
  - 平滑CE: 允许model只保持90%确定，10%留给其他词 → 泛化更好

【训练干预】
  完整的交互式干预接口，可以随时暂停训练、查看/修改权重、
  调整学习率、提升特定样本权重等。

================================================================================
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import config
from model import Seq2SeqTransformer


# ==============================================================================
# 1. 学习率调度器
# ==============================================================================
# 为什么要精心设计学习率？
#   - 太大 → 训练震荡或发散
#   - 太小 → 收敛太慢
#   - Transformer对学习率特别敏感，需要预热机制
#
# 预热 (Warmup): 前 warmup_steps 步线性增长学习率
#   原因: 训练初期模型权重随机，梯度方向不稳定
#         小学习率让模型先"找到方向"，再加速
#
# 余弦衰减 (Cosine Decay): 预热结束后余弦退火到0
#   原因: 训练后期接近最优解，需要精细调整
#         余弦衰减比线性衰减更平滑（开始慢、中间快、最后慢）
# ==============================================================================

def create_lr_scheduler(optimizer, d_model: int, warmup_steps: int, total_steps: int):
    """
    【创建学习率调度器：Warmup + Cosine Decay】

    【返回一个 LambdaLR 调度器】，每个 step 调用 scheduler.step() 更新 LR。

    【两阶段学习率曲线】
      阶段1 (step < warmup_steps):  线性预热
        lr = base_lr * step / warmup_steps
        从 0 线性增长到 base_lr

      阶段2 (step >= warmup_steps): 余弦衰减
        progress = (step - warmup) / (total - warmup)
        lr = base_lr * 0.5 * (1 + cos(π * progress))
        从 base_lr 平滑衰减到 0

    【学习率曲线示意图】
      lr
       │     ╱╲
       │    ╱  ╲
       │   ╱    ╲
       │  ╱      ╲_________
       │ ╱                  ╲___
       └─────────────────────────→ step
           ←warmup→←cosine decay→

    【参数说明】
      optimizer: AdamW优化器
      d_model: 模型维度（保留参数以兼容论文公式，实际此处未使用）
      warmup_steps: 预热步数（例如 2000步 ≈ 3个epoch）
      total_steps: 总训练步数 = epochs × batches_per_epoch

    【返回值】
      LambdaLR 调度器对象，scheduler.step() 每次调用更新学习率
    """
    def lr_lambda(step):
        """内部函数：给定step返回学习率缩放因子"""
        if step < warmup_steps:
            # 阶段1：线性预热
            # step从0到warmup_steps → 返回值从0到1
            return float(step) / float(max(1, warmup_steps))
        else:
            # 阶段2：余弦衰减
            # progress: 从预热结束到总步数的进度 (0→1)
            progress = float(step - warmup_steps) / \
                       float(max(1, total_steps - warmup_steps))
            # cos(π * progress): 从1到-1
            # 0.5 * (1 + cos): 从1到0（平滑衰减）
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 2. 标签平滑损失函数
# ==============================================================================
# 标准CE: 真实标签的one-hot向量 × 模型预测的log概率
#   Loss = -Σ y_onehot · log(p) = -log(p[true_token])
#   问题: 强迫模型100%确定 → 容易过拟合，对错误标签过于敏感
#
# Label Smoothing CE:
#   y_smooth = (1-ε) × y_onehot + ε/V × 1  (均匀分布)
#   其中 ε=0.1, V=vocab_size
#   效果: 模型不需要100%确定 → 提高泛化，让预测分布更平滑
# ==============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    标签平滑交叉熵损失 — 比标准CE更robust。

    【数学定义】
      对每个位置 i:
        nll_loss  = -log(p[true_token])           ← 标准负对数似然
        smooth    = -mean(log(p[all_tokens]))      ← 所有token的均匀损失
        loss = (1-ε) × nll_loss + ε × smooth       ← 加权组合

    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        """
        【参数说明】
          vocab_size: 词表大小（用于计算均匀分布的概率质量）
          smoothing: 平滑系数 ε（0=标准CE, 0.1=10%概率给其他词）
          ignore_index: 忽略的token ID（PAD=0）
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

        # 真实标签的置信度 = 1 - 平滑量
        # 例如 1.0 - 0.1 = 0.9 → 90%概率给正确答案
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        【前向传播 — 计算标签平滑损失】

        【参数说明】
          logits: (B, T, V) 模型输出（未归一化的分数）
          target: (B, T) 真实token ID

        【计算流程】
          1. reshape: (B,T,V) → (B*T, V)，压平batch和序列维度
          2. nll: 只取正确答案位置的对数概率
          3. smooth: 所有词的平均对数概率
          4. 忽略 PAD 位置：loss只计算有效token位置
          5. 返回标量（每个有效token的平均loss）

        【返回】
          标量 loss (单个浮点数)
        """
        B, T, V = logits.shape

        # ── 步骤1: 压平 batch 和序列维度 ──
        # (B, T, V) → (B*T, V)
        logits = logits.reshape(-1, V)
        target = target.reshape(-1)   # (B*T,)

        # ── 步骤2: 计算 log-softmax ──
        # F.log_softmax 比 log(F.softmax) 数值更稳定
        # 包含 log-sum-exp 操作，防止指数溢出
        nll = F.log_softmax(logits, dim=-1)   # (B*T, V)

        # ── 步骤3: 正确token的负对数似然 ──
        # gather(1, target.unsqueeze(1)): 从每行取出正确token的log概率
        # 然后 squeeze 掉多余维度
        nll_loss = -nll.gather(1, target.unsqueeze(1)).squeeze(1)   # (B*T,)

        # ── 步骤4: 平滑项（均匀分布损失） ──
        # 对所有词的平均 -log(p)，即均匀分布的交叉熵
        smooth_loss = -nll.mean(dim=-1)   # (B*T,)

        # ── 步骤5: 组合两项 ──
        # 90%权重给正确答案，10%给均匀分布
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # ── 步骤6: 忽略 PAD 位置 ──
        # 把 PAD 位置的 loss 置0，然后按有效token数平均
        mask = (target != self.ignore_index).float()   # (B*T,)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        return loss


# ==============================================================================
# 3. 训练干预接口
# ==============================================================================
# 让你在训练过程中"停下来看看"——这是理解 Transformer 训练动态的关键工具。
# ==============================================================================

class TrainingIntervention:
    """
    训练干预接口 —— 让你在训练中手动调整各种参数。

    【使用场景】
      1. 想看看某个权重矩阵长什么样 → show weights
      2. 想看看梯度是否在正常流动 → show grad
      3. 训练一段时间后loss不再下降 → 手动降LR
      4. 某些句子翻译总是很差 → boost它们
      5. 怀疑模型出现了问题 → pause, 检查, 然后继续

    【交互命令】
      show weights <name>  - 查看权重统计
      show grad <name>     - 查看梯度
      set lr <value>       - 手动设置学习率
      boost <idx> <w>      - 提升特定样本权重
      boost_text <txt> <w> - 文本匹配提升权重
      unboost              - 清除所有样本权重调整
      stats                - 当前训练统计
      save <path>          - 保存模型
      continue             - 继续训练
      help                 - 帮助
    """

    def __init__(self):
        self.paused = False                # 是否暂停
        self.pause_after_epoch = False     # epoch结束后自动暂停
        self.sample_boosts = {}            # {dataset_idx: weight_multiplier}
        self.text_boosts = []              # [(text_pattern, weight), ...]
        self.custom_lr = None              # 手动设置的LR
        self.epoch_stats = []              # [{epoch, loss, val_loss, lr, time}, ...]

    def pause(self):
        """暂停训练（在下一个batch开始时进入交互模式）"""
        self.paused = True

    def boost_sample(self, idx: int, weight: float = 5.0):
        """
        【提升特定样本权重】
        让某个样本的loss贡献放大weight倍。
        相当于告诉模型："请特别关注这个样本！"

        【使用示例】
          intervention.boost_sample(42, weight=10.0)
          # 第42个样本的loss被放大10倍
          # 模型会更多地调整参数以适应这个样本
        """
        self.sample_boosts[idx] = weight
        print(f"[Intervention] 样本 {idx} 权重设为 {weight:.1f}x")

    def boost_sentence(self, pattern: str, weight: float = 5.0):
        """
        【文本匹配提升权重】
        对包含特定文本的所有样本加权。
        比如想让模型把"深度学习"翻译得更好。

        【使用示例】
          intervention.boost_sentence("深度学习", weight=5.0)
          # 所有包含"深度学习"的句子loss放大5倍
        """
        self.text_boosts.append((pattern, weight))
        print(f"[Intervention] 包含 '{pattern}' 的样本权重设为 {weight:.1f}x")

    def clear_boosts(self):
        """清除所有样本权重调整"""
        self.sample_boosts.clear()
        self.text_boosts.clear()
        print(f"[Intervention] 已清除所有样本权重调整")

    def interactive_mode(self, model, optimizer, scheduler, epoch, batch_idx, loss):
        """
        【交互模式 REPL】
        训练暂停时进入，提供命令行界面检查和修改训练状态。

        这个REPL让你在训练过程中：
          - 查看任意层的权重和梯度
          - 手动调整学习率
          - 对特定样本加权
          - 保存中间检查点
        """
        print("\n" + "=" * 60)
        print("  [训练干预模式] — 输入 help 查看命令")
        print("=" * 60)

        while True:
            try:
                cmd = input(" 干预> ").strip()
            except (EOFError, KeyboardInterrupt):
                cmd = "continue"

            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            # ── 继续训练 ──
            if action in ("continue", "c"):
                self.paused = False
                break

            # ── 帮助 ──
            elif action in ("help", "h"):
                print("  命令列表:")
                print("    weights <name>     - 查看权重矩阵 (如: weights embedding)")
                print("    grad <name>        - 查看梯度")
                print("    lr <value>         - 设置学习率 (如: lr 5e-5)")
                print("    boost <idx> <w>    - 提升样本权重")
                print("    boost_text <t> <w> - 文本匹配提升权重")
                print("    stats              - 训练统计")
                print("    save <path>        - 保存模型")
                print("    continue           - 继续训练")

            # ── 查看权重 ──
            elif action in ("weights", "w"):
                if len(parts) < 2:
                    # 无参数：列出所有参数名称
                    for name, param in model.named_parameters():
                        print(f"    {name}: {list(param.shape)}")
                else:
                    name = parts[1]
                    found = False
                    for pname, param in model.named_parameters():
                        if name in pname:
                            print(f"\n  [{pname}] shape={list(param.shape)}")
                            print(f"    mean={param.data.mean():.6f}  "
                                  f"std={param.data.std():.6f}")
                            print(f"    min={param.data.min():.6f}   "
                                  f"max={param.data.max():.6f}")
                            # 前10个值
                            flat = param.data.flatten()[:10]
                            print(f"    前10个值: {flat.tolist()}")
                            found = True
                            break
                    if not found:
                        print(f"  未找到参数: {name}")

            # ── 查看梯度 ──
            elif action in ("grad", "g"):
                if len(parts) < 2:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gnorm = param.grad.norm().item()
                            print(f"    {name}: grad_norm={gnorm:.6f}")
                else:
                    name = parts[1]
                    for pname, param in model.named_parameters():
                        if name in pname and param.grad is not None:
                            print(f"\n  [{pname}] grad={list(param.grad.shape)}")
                            print(f"    grad_mean={param.grad.mean():.6f}  "
                                  f"std={param.grad.std():.6f}")
                            print(f"    grad_max={param.grad.max():.6f}")
                            break
                    else:
                        print(f"  未找到参数或其梯度: {name}")

            # ── 设置学习率 ──
            elif action == "lr":
                if len(parts) >= 2:
                    new_lr = float(parts[1])
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    self.custom_lr = new_lr
                    print(f"  学习率已设置为: {new_lr:.2e}")

            # ── 样本加权 ──
            elif action == "boost":
                if len(parts) >= 3:
                    idx = int(parts[1])
                    w = float(parts[2])
                    self.boost_sample(idx, w)

            elif action == "boost_text":
                if len(parts) >= 3:
                    text = parts[1]
                    w = float(parts[2])
                    self.boost_sentence(text, w)

            elif action == "unboost":
                self.clear_boosts()

            # ── 训练统计 ──
            elif action == "stats":
                print(f"  Epoch: {epoch} | Batch: {batch_idx}")
                print(f"  Loss: {loss:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                print(f"  样本权重提升: {len(self.sample_boosts)} 个")
                for stat in self.epoch_stats[-3:]:
                    print(f"  E{stat['epoch']:2d}: "
                          f"train={stat['loss']:.4f} "
                          f"val={stat.get('val_loss',0):.4f} "
                          f"time={stat['time']:.1f}s")

            # ── 保存 ──
            elif action == "save":
                path = parts[1] if len(parts) >= 2 \
                    else os.path.join(config.save_dir, "intervention.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, path)
                print(f"  模型已保存到: {path}")

            else:
                print(f"  未知命令: {action} (输入 help 查看帮助)")


# ==============================================================================
# 4. 训练器
# ==============================================================================

class Trainer:
    """
    Transformer 训练器 — 串联所有训练组件。

    【职责】
      - 管理模型、优化器、调度器、损失函数的生命周期
      - 执行 Teacher Forcing 训练循环
      - 实时日志（loss、lr、梯度norm、耗时）
      - 验证集评估
      - 自动保存最佳模型
      - 提供干预接口
    """

    def __init__(self, model: Seq2SeqTransformer, tokenizer=None):
        """
        【初始化训练器】
        创建优化器、损失函数，把模型移到GPU。

        【参数说明】
          model: 创建好的 Seq2SeqTransformer 实例
          tokenizer: 分词器（保留参数，可能用于评估）
        """
        # ── 模型移到 GPU ──
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.device = config.device

        # ── 损失函数 ──
        # Label Smoothing 让模型不"过自信"
        self.criterion = LabelSmoothingLoss(
            vocab_size=config.vocab_size,
            smoothing=config.label_smoothing,  # 0.1
            ignore_index=0,                      # PAD
        )

        # ── AdamW 优化器 ──
        # AdamW 和 Adam 的区别：
        #   Adam:  L2正则化通过修改梯度实现（耦合在momentum中）
        #   AdamW: 权重衰减直接加到权重上（解耦）
        #   AdamW在实践中效果更好，是Transformer训练的标配
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,               # 初始学习率 1e-4
            betas=(config.adam_beta1,   # β1=0.9
                   config.adam_beta2),  # β2=0.98 (Transformer常用)
            eps=config.adam_eps,        # 1e-9
            weight_decay=config.weight_decay,  # 0.01 (权重衰减)
        )

        # 调度器在 fit() 中创建（需要知道 total_steps）
        self.scheduler = None

        # ── 训练干预 ──
        self.intervention = TrainingIntervention()

        # ── 统计变量 ──
        self.train_losses = []        # 每个epoch的平均训练loss
        self.val_losses = []          # 每个epoch的验证loss
        self.best_val_loss = float("inf")   # 最佳验证loss
        self.total_train_time = 0.0         # 总训练时间（秒）
        self.current_epoch = 0              # 当前epoch编号
        self.global_step = 0                # 当前全局步数

        # 确保保存目录存在
        os.makedirs(config.save_dir, exist_ok=True)

        print(f"\n[Trainer] 初始化完成")
        print(f"  设备: {self.device}")
        print(f"  优化器: AdamW (lr={config.lr}, "
              f"β1={config.adam_beta1}, β2={config.adam_beta2})")
        print(f"  损失函数: LabelSmoothing (smoothing={config.label_smoothing})")
        print(f"  梯度裁剪: {config.grad_clip}")

    # ======================================================================
    # 4a. 训练一个Epoch
    # ======================================================================

    def train_epoch(self, train_loader, epoch: int) -> float:
        """
        【训练一个完整的 epoch】

        每个 epoch 包含：
          遍历全部训练数据一次
          对每个batch做一次前向+反向+优化
          每N个batch打印一次日志
          支持中途暂停干预

        【Teacher Forcing 详解】
          batch 中的数据：
            src:     [SOS, 中文tokens..., EOS, PAD...]
            tgt_in:  [SOS, 英文tokens..., PAD...]      ← 去掉最后的EOS
            tgt_out: [英文tokens..., EOS, PAD...]        ← 去掉最前的SOS

          模型输入:  src + tgt_in
          模型输出:  logits (预测的下一个token)
          loss:      logits vs tgt_out (真实的下一个token)

        【返回】
          (avg_loss, epoch_time) 平均loss和epoch耗时
        """
        # 切换到训练模式（启用 Dropout）
        self.model.train()

        total_loss = 0.0
        total_tokens = 0           # 有效token总数（排除PAD）
        epoch_start = time.time()

        for batch_idx, (src, src_mask, tgt_in, tgt_mask, tgt_out) \
                in enumerate(train_loader):

            # ── 检查是否需要暂停（干预模式） ──
            if self.intervention.paused:
                self.intervention.interactive_mode(
                    self.model, self.optimizer, self.scheduler,
                    epoch, batch_idx,
                    total_loss / max(1, batch_idx),
                )

            # ── 数据移到 GPU ──
            # non_blocking=True: 异步传输，CPU继续做其他事
            src = src.to(self.device)                    # (B, S)
            src_mask = src_mask.to(self.device)          # (B, S)
            tgt_in = tgt_in.to(self.device)              # (B, T)
            tgt_mask = tgt_mask.to(self.device)          # (B, T)
            tgt_out = tgt_out.to(self.device)            # (B, T)

            # ── 前向传播 (Teacher Forcing) ──
            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            # logits: (B, T, V) 每个位置对词表中所有token的分数

            # ── 计算损失 ──
            loss = self.criterion(logits, tgt_out)

            # ── 反向传播 ──
            # 三步标准流程：
            #   1. zero_grad(): 清零梯度缓存（否则会累积）
            #   2. loss.backward(): 计算梯度（自动微分）
            #   3. optimizer.step(): 用梯度更新参数
            self.optimizer.zero_grad()
            loss.backward()

            # ── 梯度裁剪 ──
            # 限制梯度的L2范数不超过 grad_clip (1.0)
            # 为什么？防止某个batch的梯度特别大导致训练不稳定
            # RNN/Transformer中常见问题：长序列梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.grad_clip,   # 1.0
            )

            self.optimizer.step()

            # ── 更新学习率（每个step更新一次） ──
            if self.scheduler:
                self.scheduler.step()

            # ── 统计 ──
            total_loss += loss.item()
            total_tokens += tgt_mask.sum().item()
            self.global_step += 1

            # ── 实时日志 ──
            if batch_idx % config.log_interval == 0 \
                    or batch_idx == len(train_loader) - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                print(
                    f"  [Epoch {epoch:3d}] "
                    f"Batch {batch_idx:4d}/{len(train_loader):4d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Grad Norm: {grad_norm:.2f} | "
                    f"{elapsed:.1f}s"
                )

        # epoch 结束
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        return avg_loss, epoch_time

    # ======================================================================
    # 4b. 验证
    # ======================================================================

    @torch.no_grad()   # 禁用梯度计算（节省显存，加速推理）
    def validate(self, val_loader) -> float:
        """
        【在验证集上评估模型】

        和训练的区别：
          1. model.eval() 关闭 Dropout（所有神经元都参与）
          2. @torch.no_grad() 不计算梯度
          3. 不调用 backward() 和 optimizer.step()
          4. 同样的 Teacher Forcing 方式（用真实标签，不用预测值）

        【返回】
          平均验证 loss（越低越好）
        """
        self.model.eval()
        total_loss = 0.0

        for src, src_mask, tgt_in, tgt_mask, tgt_out in val_loader:
            src = src.to(self.device)
            src_mask = src_mask.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            tgt_out = tgt_out.to(self.device)

            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            loss = self.criterion(logits, tgt_out)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    # ======================================================================
    # 4c. 完整训练循环
    # ======================================================================

    def fit(self, train_loader, val_loader, epochs: int = None):
        """
        【完整训练循环 — 运行多个 epoch】

        【流程】
          for epoch in 1..epochs:
            1. train_epoch() — 遍历全部训练数据
            2. validate()    — 验证集评估
            3. 打印总结
            4. 保存最佳模型
            5. （可选）epoch结束干预

        【参数说明】
          train_loader: 训练 DataLoader
          val_loader: 验证 DataLoader
          epochs: 覆盖config.epochs（默认None → 用config的值）
        """
        epochs = epochs or config.epochs

        # 总训练步数 = epoch数 × 每个epoch的batch数
        total_steps = epochs * len(train_loader)

        # ── 创建学习率调度器 ──
        self.scheduler = create_lr_scheduler(
            self.optimizer,
            d_model=config.d_model,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
        )

        print("\n" + "=" * 60)
        print(f"  开始训练: {epochs} epochs, "
              f"{len(train_loader)} batches/epoch")
        print(f"  总步数: {total_steps}, "
              f"预热步数: {config.warmup_steps}")
        print(f"  模型参数: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60 + "\n")

        train_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # ── 训练一个 epoch ──
            train_loss, epoch_time = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # ── 验证 ──
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.total_train_time += epoch_time

            # ── Epoch 总结 ──
            total_elapsed = time.time() - train_start
            lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n  ═══ Epoch {epoch:3d}/{epochs} 总结 ═══")
            print(f"  训练 Loss: {train_loss:.4f} | "
                  f"验证 Loss: {val_loss:.4f}")
            print(f"  学习率:     {lr:.2e}")
            print(f"  本 Epoch 耗时: {epoch_time:.1f}s | "
                  f"总耗时: {total_elapsed:.1f}s")
            print(f"  预计剩余:      "
                  f"{(total_elapsed / epoch) * (epochs - epoch):.0f}s")
            print()

            # ── 保存最佳模型 ──
            # 只在验证loss改善时保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if config.save_best:
                    self.save("best_model.pt")
                    print(f"  [BEST] 最佳模型已保存 "
                          f"(val_loss={val_loss:.4f})")

            # ── 定期保存检查点 ──
            if epoch % 10 == 0:
                self.save(f"checkpoint_epoch{epoch}.pt")

            # ── Epoch结束干预 ──
            if self.intervention.pause_after_epoch:
                self.intervention.paused = True
                self.intervention.interactive_mode(
                    self.model, self.optimizer, self.scheduler,
                    epoch, -1, train_loss,
                )

            # ── 记录统计 ──
            self.intervention.epoch_stats.append({
                "epoch": epoch,
                "loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "time": epoch_time,
            })

        # ── 训练完成 ──
        self.total_train_time = time.time() - train_start
        print("=" * 60)
        print(f"  训练完成!")
        print(f"  总耗时: {self.total_train_time:.1f}s "
              f"({self.total_train_time/60:.1f}min)")
        print(f"  最佳验证 Loss: {self.best_val_loss:.4f}")
        print("=" * 60)

    # ======================================================================
    # 4d. 保存与加载
    # ======================================================================

    def save(self, filename: str):
        """
        【保存训练状态到文件】

        保存的内容不仅是模型权重，还包括：
          - optimizer 状态（训练可恢复）
          - epoch 编号（继续训练时知道从哪里开始）
          - loss 历史（画图分析）
          - 最佳 loss（追踪改进）

        保存的文件可以用 torch.load() 加载，用 load() 恢复训练。
        """
        path = os.path.join(config.save_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
        torch.save(checkpoint, path)
        print(f"  [保存] 模型已保存到: {path}")
        return path

    def load(self, filename: str):
        """
        【从文件恢复训练状态】

        恢复后可以：
          1. 继续训练（optimizer和学习率也恢复了）
          2. 评估（验证loss历史可用于分析）
          3. 推理（只用模型权重）
        """
        path = os.path.join(config.save_dir, filename)
        if not os.path.exists(path):
            print(f"  [加载] 文件不存在: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"  [加载] 模型已从 {path} 加载 "
              f"(epoch={self.current_epoch})")
        return True


# ==============================================================================
# 5. 快捷训练函数
# ==============================================================================

def train_model(train_loader, val_loader, tokenizer, resume_from: str = None):
    """
    【创建模型 → 训练 → 保存】
    这是 main.py 调用的入口函数。
    如果需要恢复训练，传入 resume_from 参数（检查点文件名）。
    """
    # ── 创建模型 ──
    model = Seq2SeqTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_encoder_layers=config.n_encoder_layers,
        n_decoder_layers=config.n_decoder_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        max_len=config.max_pos_len,
        pad_idx=0,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n[模型] 总参数: {param_count:,}")
    print(f"[模型] 可训练参数: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── 创建训练器 ──
    trainer = Trainer(model, tokenizer)

    # ── 恢复训练（如果指定） ──
    if resume_from:
        trainer.load(resume_from)

    # ── 开始训练 ──
    trainer.fit(train_loader, val_loader)

    # ── 保存最终模型 ──
    trainer.save("final_model.pt")

    return trainer, model
