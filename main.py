"""
================================================================================
Transformer 教育版 — 主入口 (main.py)
=======================================
命令行统一入口，三条命令覆盖完整工作流。

【用法速查】
  python main.py train              ← 从头训练模型
  python main.py train --resume     ← 从检查点恢复训练
  python main.py infer              ← 交互式翻译对话
  python main.py infer "你好"       ← 单句翻译
  python main.py infer "你好" --weight  ← 单句翻译 + 注意力可视化
  python main.py test               ← 测试所有模块是否正常

================================================================================
"""

import os
import sys
import argparse

# 确保当前目录在搜索路径中（方便 import 同目录模块）
sys.path.insert(0, os.path.dirname(__file__))

from config import config
from data import prepare_data
from tokenizer import SimpleTokenizer
from model import Seq2SeqTransformer
from train import train_model         # Trainer + 训练循环
from inference import interactive_inference, single_translate


# ==============================================================================
# 训练命令
# ==============================================================================

def cmd_train(resume: bool = False):
    """
    【训练模式 — 启动或恢复训练】

    【执行流程】
      1. 打印训练配置（确认参数正确）
      2. prepare_data() → 加载数据 + 构建词表 + 创建DataLoader
      3. train_model() → 创建模型 → Trainer.fit() → 保存最佳模型

    【恢复训练】
      --resume 参数从最近的检查点继续训练。
      恢复时会加载：
        - 模型权重（继续用学到的参数）
        - 优化器状态（保持momentum等历史信息）
        - epoch编号（从上次结束的位置继续）
    """
    print("\n" + "=" * 50)
    print("  Transformer 教育版 - 训练模式")
    print("  中文 → 英文 翻译")
    print("=" * 50 + "\n")

    # ── 打印配置 ──
    print("=" * 60)
    print("  训练配置:")
    print(f"    模型维度:    d_model={config.d_model}, heads={config.n_heads}")
    print(f"    层数:        encoder={config.n_encoder_layers}, "
          f"decoder={config.n_decoder_layers}")
    print(f"    FFN维度:     d_ff={config.d_ff}")
    print(f"    词表大小:    {config.vocab_size}")
    print(f"    序列长度:    {config.max_seq_len}")
    print(f"    批次大小:    {config.batch_size}")
    print(f"    训练轮数:    {config.epochs}")
    print(f"    学习率:      {config.lr}")
    print(f"    Dropout:     {config.dropout}")
    print(f"    训练样本:    {config.max_train_samples}")
    print(f"    设备:        {config.device}")
    print("=" * 60 + "\n")

    # 1. 准备数据
    train_loader, val_loader, tokenizer = prepare_data()

    # 2. 训练
    resume_from = "best_model.pt" if resume else None
    trainer, model = train_model(train_loader, val_loader,
                                  tokenizer, resume_from=resume_from)

    print("\n训练完成！运行 `python main.py infer` 开始翻译。")
    return trainer, model


# ==============================================================================
# 推理命令
# ==============================================================================

def cmd_infer(text: str = None, show_weights: bool = False):
    """
    【推理模式 — 加载模型进行翻译】

    无参数 → 交互模式（REPL对话）
    有参数 → 单句翻译

    --weight → 额外打印注意力权重热力图
    """
    model_path = os.path.join(config.save_dir, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"[错误] 未找到模型文件: {model_path}")
        print("请先运行训练: python main.py train")
        return

    if text:
        # 单句翻译
        single_translate(model_path, config.tokenizer_path,
                         text, show_weights=show_weights)
    else:
        # 交互模式
        interactive_inference(model_path, config.tokenizer_path)


# ==============================================================================
# 测试命令
# ==============================================================================

def cmd_test():
    """
    【模块测试 — 验证所有组件可以正常运行】

    测试覆盖:
      1. 分词器: 构建词表 → 编码 → 解码
      2. 模型: 创建 → 前向传播（verbose）
      3. Mask: 因果mask 和 padding mask
      4. Loss: Label Smoothing 计算
      5. 推理: 端到端翻译（未训练的模型，输出随机）

    如果所有测试通过 → 系统可正常训练
    """
    print("\n" + "---" * 20)
    print("  Transformer 教育版 - 模块测试")
    print("---" * 20 + "\n")

    import torch

    # ── 测试1: 分词器 ──
    print("=" * 50)
    print("  测试 1: 分词器")
    print("=" * 50)
    tokenizer = SimpleTokenizer(vocab_size=1000)
    zh_texts = ["你好世界", "今天天气很好", "机器学习很有趣"]
    en_texts = ["hello world", "the weather is nice today",
                "machine learning is fun"]
    tokenizer.build_vocab(zh_texts, en_texts)
    print(f"  词表大小: {len(tokenizer)}")

    zh_ids = tokenizer.encode_zh("你好世界")
    en_ids = tokenizer.encode_en("hello world")
    print(f"  '你好世界' → {zh_ids}")
    print(f"  'hello world' → {en_ids}")
    print(f"  解码 → '{tokenizer.decode(en_ids)}'")

    # ── 测试2: 模型构建 ──
    print(f"\n{'='*50}")
    print(f"  测试 2: 模型构建")
    print(f"{'='*50}")
    model = Seq2SeqTransformer(
        vocab_size=1000, d_model=128, n_heads=4,
        n_encoder_layers=2, n_decoder_layers=2,
        d_ff=256, dropout=0.1, max_len=32, pad_idx=0,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数数量: {param_count:,}")

    # ── 测试3: 前向传播 (verbose) ──
    print(f"\n{'='*50}")
    print(f"  测试 3: 前向传播 (Verbose)")
    print(f"{'='*50}")

    B, S, T = 2, 8, 6   # batch=2, src_len=8, tgt_len=6
    src = torch.randint(4, 100, (B, S))       # 随机token ID [4,100)
    tgt_in = torch.randint(4, 100, (B, T))
    src_mask = torch.ones(B, S, dtype=torch.bool)
    tgt_mask = torch.ones(B, T, dtype=torch.bool)

    logits = model(src, tgt_in, src_mask, tgt_mask, verbose=True)
    print(f"\n  输出形状: {logits.shape}  (预期: ({B}, {T}, 1000))")

    # ── 测试4: Mask创建 ──
    print(f"\n{'='*50}")
    print(f"  测试 4: Mask 创建")
    print(f"{'='*50}")
    causal = model.create_causal_mask(5, torch.device("cpu"))
    print(f"  因果 mask (5x5): 0=可见, -inf=屏蔽")
    print(causal.squeeze())

    # ── 测试5: Label Smoothing ──
    print(f"\n{'='*50}")
    print(f"  测试 5: Label Smoothing 损失")
    print(f"{'='*50}")
    from train import LabelSmoothingLoss
    criterion = LabelSmoothingLoss(vocab_size=1000, smoothing=0.1)
    tgt_out = torch.randint(4, 100, (B, T))
    loss = criterion(logits, tgt_out)
    print(f"  Loss (smoothing=0.1): {loss.item():.4f}")

    # ── 测试6: 推理 ──
    print(f"\n{'='*50}")
    print(f"  测试 6: 推理")
    print(f"{'='*50}")
    from inference import Translator
    translator = Translator(model, tokenizer, device="cpu", max_len=32)
    result = translator.translate("你好", show_weights=True)
    print(f"  翻译 '你好' → '{result['translation']}'")
    print(f"  生成 token: {result['tokens']}")

    print(f"\n{'='*50}")
    print(f"  [OK] 所有测试通过!")
    print(f"{'='*50}")


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """
    【程序入口 — 解析命令行参数并分发到对应子命令】

    使用 argparse 实现子命令模式:
      python main.py train     → cmd_train()
      python main.py infer     → cmd_infer()
      python main.py test      → cmd_test()
    """
    parser = argparse.ArgumentParser(
        description="Transformer 教育版 — 中文→英文翻译",
    )

    # ── 子命令 ──
    subparsers = parser.add_subparsers(
        dest="command", help="子命令")

    # train 子命令
    p_train = subparsers.add_parser("train", help="训练模型")
    p_train.add_argument("--resume", action="store_true",
                         help="从检查点恢复训练")

    # infer 子命令
    p_infer = subparsers.add_parser("infer", help="翻译推理")
    p_infer.add_argument("text", nargs="?", default=None,
                         help="要翻译的中文文本（不填则进入交互模式）")
    p_infer.add_argument("--weight", action="store_true",
                         help="显示注意力权重")

    # test 子命令
    subparsers.add_parser("test", help="测试所有模块")

    # ── 解析 ──
    args = parser.parse_args()

    # ── 分发 ──
    if args.command == "train":
        cmd_train(resume=args.resume)
    elif args.command == "infer":
        cmd_infer(text=args.text, show_weights=args.weight)
    elif args.command == "test":
        cmd_test()
    else:
        # 没有子命令 → 打印帮助
        parser.print_help()
        print("\n示例:")
        print("  python main.py train")
        print("  python main.py infer")
        print("  python main.py infer \"你好世界\"")
        print("  python main.py infer \"你好世界\" --weight")
        print("  python main.py test")


if __name__ == "__main__":
    main()
