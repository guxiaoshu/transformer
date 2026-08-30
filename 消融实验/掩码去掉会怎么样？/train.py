
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




def create_lr_scheduler(optimizer, d_model: int, warmup_steps: int, total_steps: int):

    def lr_lambda(step):

        if step < warmup_steps:
            # 线性预热

            return float(step) / float(max(1, warmup_steps))
        else:
            # ：余弦衰减

            progress = float(step - warmup_steps) / \
                       float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
# 也不懂为什么要这样子设定，照做就行


# 标签平滑损失函数

#   Loss = -Σ y_onehot · log(p) = -log(p[true_token])
#强迫模型100%确定 容易过拟合

# Label Smoothing CE:
#   y_smooth = (1-ε) × y_onehot + ε/V × 1  (均匀分布)
#    ε=0.1, V=vocab_size
#   模型不需要100%确定 提高泛化


class LabelSmoothingLoss(nn.Module):
#最重要的loss

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):

        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index

        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        B, T, V = logits.shape

        # 压平 batch 和序列维度 
        # (B, T, V) -> (B*T, V)
        logits = logits.reshape(-1, V)
        target = target.reshape(-1)   # (B*T,)

        # 计算 log-softmax 

        nll = F.log_softmax(logits, dim=-1)   # (B*T, V)

        # 正确token的负对数似然
        # gather(1, target.unsqueeze(1)): 从每行取出正确token的log概率
        # 然后 squeeze 掉多余维度
        nll_loss = -nll.gather(1, target.unsqueeze(1)).squeeze(1)   # (B*T,)

        # 对所有词的平均 -log(p)，即均匀分布的交叉熵
        smooth_loss = -nll.mean(dim=-1)   # (B*T,)

        # 组合两项 
        # 90%权重给正确答案，10%给均匀分布
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        # 忽略 PAD 位置 
        # 把 PAD 位置的 loss 置0，然后按有效token数平均
        mask = (target != self.ignore_index).float()   # (B*T,)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1)

        return loss






class TrainingIntervention:
    # 训练干预


    def __init__(self):
        self.paused = False                # 是否暂停
        self.pause_after_epoch = False     # epoch结束后自动暂停
        self.sample_boosts = {}            # {dataset_idx: weight_multiplier}
        self.text_boosts = []              # [(text_pattern, weight), ...]
        self.custom_lr = None              # 手动设置的LR
        self.epoch_stats = []              # [{epoch, loss, val_loss, lr, time}, ...]

    def pause(self):
     #暂停    
        self.paused = True






    def interactive_mode(self, model, optimizer, scheduler, epoch, batch_idx, loss):

        #训练暂停时进入，提供命令行界面检查和修改训练状态。


        print("\n" + "=" * 60)
        print(" [训练干预输入 help 查看命令")
        print("=" * 60)

        while True:
            try:
                cmd = input(" 干预 ").strip()
            except (EOFError, KeyboardInterrupt):
                cmd = "continue"

            if not cmd:
                continue

            parts = cmd.split()
            action = parts[0].lower()

            #  继续训练 
            if action in ("continue", "c"):
                self.paused = False
                break

            # ── 帮助 ──
            elif action in ("help", "h"):
                print("    weights    查看权重矩阵 ")
                print("    grad        查看梯度")
                print("    lr        手动设置学习率")
                print("    stats          训练统计")
                print("    save     保存模型")
                print("    continue        继续训练")

            # ── 查看权重 ──
            elif action in ("weights", "w"):
                if len(parts) < 2:

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
                        print(f"  没找到参数 {name}")

            # 查看梯度 
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
                        print(f"  没找到参数 {name}")

            #  设置学习率
            elif action == "lr":
                if len(parts) >= 2:
                    new_lr = float(parts[1])
                    for g in optimizer.param_groups:
                        g["lr"] = new_lr
                    self.custom_lr = new_lr
                    print(f"  学习率已设置为: {new_lr:.2e}")


            elif action == "boost_text":
                if len(parts) >= 3:
                    text = parts[1]
                    w = float(parts[2])
                    self.boost_sentence(text, w)

            elif action == "unboost":
                self.clear_boosts()



            # 保存 
            elif action == "save":
                path = parts[1] if len(parts) >= 2 \
                    else os.path.join(config.save_dir, "intervention.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                }, path)
                print(f"  模型保存到{path}")

            else:
                print(f"  不要输入未知命令 {action} ")




class Trainer:


    def __init__(self, model: Seq2SeqTransformer, tokenizer=None):

        # 模型移到 GPU 
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.device = config.device

        #  Label Smoothing 函数 
        self.criterion = LabelSmoothingLoss(
            vocab_size=config.vocab_size,
            smoothing=config.label_smoothing,  # 0.1
            ignore_index=0,                      # PAD
        )

        # AdamW 优化器 
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.lr,               # 初始学习率 1e-4
            betas=(config.adam_beta1,   # β1=0.9
                   config.adam_beta2),  # β2=0.98 (Transformer常用)
            eps=config.adam_eps,        # 1e-9
            weight_decay=config.weight_decay,  # 0.01 (权重衰减)
        )


        self.scheduler = None

        # 训练干预 
        self.intervention = TrainingIntervention()

        # 统计变量
        self.train_losses = []        # 每个epoch的平均训练loss
        self.val_losses = []          # 每个epoch的验证loss
        self.best_val_loss = float("inf")   # 最佳验证loss
        self.total_train_time = 0.0         # 总训练时间（秒）
        self.current_epoch = 0              # 当前epoch编号
        self.global_step = 0                # 当前全局步数


        os.makedirs(config.save_dir, exist_ok=True)


    # 训练一个Epoch

    def train_epoch(self, train_loader, epoch: int) -> float:

        self.model.train()

        total_loss = 0.0
        total_tokens = 0           # 有效token总数
        epoch_start = time.time()

        for batch_idx, (src, src_mask, tgt_in, tgt_mask, tgt_out) \
                in enumerate(train_loader):

            #  检查是否需要暂停
            if self.intervention.paused:
                self.intervention.interactive_mode(
                    self.model, self.optimizer, self.scheduler,
                    epoch, batch_idx,
                    total_loss / max(1, batch_idx),
                )

            # 数据移到 GPU
            src = src.to(self.device)                    # (B, S)
            src_mask = src_mask.to(self.device)          # (B, S)
            tgt_in = tgt_in.to(self.device)              # (B, T)
            tgt_mask = tgt_mask.to(self.device)          # (B, T)
            tgt_out = tgt_out.to(self.device)            # (B, T)

            # 前向传播
            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            # logits: (B, T, V) 每个位置对词表中所有token的分数

            # 计算损失
            loss = self.criterion(logits, tgt_out)

            # 反向传播 
            # 三步标准流程，深度学习三件套
            #   zero_grad(): 清零梯度缓存
            #   loss.backward(): 计算梯度（自动微分）
            #   optimizer.step(): 用梯度更新参数
            self.optimizer.zero_grad()
            loss.backward()

            #  梯度裁剪
            # 限制梯度的L2范数不超过 grad_clip (1.0)
            # 防止某个batch的梯度特别大导致训练不稳定
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.grad_clip,   # 1.0
            )

            self.optimizer.step()

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            #  统计
            total_loss += loss.item()
            total_tokens += tgt_mask.sum().item()
            self.global_step += 1

            #实时打印
            if batch_idx % config.log_interval == 0 \
                    or batch_idx == len(train_loader) - 1:
                lr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_start
                print(
                    f" （第{epoch:3d}个 Epoch） "#第几个epoch
                    f"Batch {batch_idx:4d}/{len(train_loader):4d} | "#第几个batch
                    f"Loss: {loss.item():.4f} | "#损失函数
                    f"LR: {lr:.2e} | "#学习率
                    f"Grad Norm: {grad_norm:.2f} | "#梯度
                    f"{elapsed:.1f}s"#用时
                )

        # epoch 结束
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        return avg_loss, epoch_time

    # 验证

    @torch.no_grad()  
    def validate(self, val_loader) -> float:

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


    # 完整训练循环


    def fit(self, train_loader, val_loader, epochs: int = None):
   
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

        print(f"  开始训练: {epochs} epochs, "
              f"{len(train_loader)} batches")
        print(f"  模型参数"
              f"{sum(p.numel() for p in self.model.parameters()):,}")
        print("=" * 60 + "\n")

        train_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # 训练一个 epoch 
            train_loss, epoch_time = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            #  验证 
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.total_train_time += epoch_time

            #  Epoch 总结
            total_elapsed = time.time() - train_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"  训练 Loss {train_loss:.4f} | "
                  f"验证 Loss {val_loss:.4f}")
            print(f"  学习率     {lr:.2e}")
            print(f"  这个 Epoch 耗时 {epoch_time:.1f}s | "
                  f"总耗时 {total_elapsed:.1f}s")


            # 保存最佳模型
            # 只在验证loss改善时保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if config.save_best:
                    self.save("best_model.pt")
                    print(f"  模型已保存 "
                          f"(val_loss={val_loss:.4f})")

            # 定期保存
            if epoch % 10 == 0:
                self.save(f"checkpoint_epoch{epoch}.pt")

            # Epoch结束干预 
            if self.intervention.pause_after_epoch:
                self.intervention.paused = True
                self.intervention.interactive_mode(
                    self.model, self.optimizer, self.scheduler,
                    epoch, -1, train_loss,
                )

            #  记录统计 
            self.intervention.epoch_stats.append({
                "epoch": epoch,
                "loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "time": epoch_time,
            })

        #  训练完成 
        self.total_train_time = time.time() - train_start
        print("=" * 60)
        print(f"  训练完成")
        print(f"  总耗时: {self.total_train_time:.1f}s "
              f"({self.total_train_time/60:.1f}min)")
        print("=" * 60)

    # 保存与加载


    def save(self, filename: str):

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
        print(f"   模型保存到 {path}")
        return path

    def load(self, filename: str):

        path = os.path.join(config.save_dir, filename)

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        return True


 
#  快捷训练函数


def train_model(train_loader, val_loader, tokenizer, resume_from: str = None):

    # 创建模型 
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


    trainer = Trainer(model, tokenizer)


    if resume_from:
        trainer.load(resume_from)

    # 开始训练
    trainer.fit(train_loader, val_loader)

    # 保存最终模型
    trainer.save("final_model.pt")

    return trainer, model
