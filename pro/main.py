
import os
import sys
import argparse


sys.path.insert(0, os.path.dirname(__file__))

from config import config
from data import prepare_data
from model import Seq2SeqTransformer
from train import train_model         # Trainer + 训练循环
from inference import interactive_inference, single_translate



# 训练命令

def cmd_train(resume: bool = False):


    print(f"    模型维度   d_model={config.d_model}, heads={config.n_heads}")
    print(f"    层数       encoder={config.n_encoder_layers}, "
          f"decoder={config.n_decoder_layers}")
    print(f"    FFN维度    d_ff={config.d_ff}")
    print(f"    词表大小{config.vocab_size}")
    print(f"    序列长度    {config.max_seq_len}")
    print(f"    批次大小    {config.batch_size}")
    print(f"    训练轮数    {config.epochs}")
    print(f"    学习率      {config.lr}")
    print(f"    Dropout    {config.dropout}")
    print(f"    训练样本    {config.max_train_samples}")
    print("=" * 60 + "\n")


    train_loader, val_loader, tokenizer = prepare_data()


    resume_from = "best_model.pt" if resume else None
    trainer, model = train_model(train_loader, val_loader,
                                  tokenizer, resume_from=resume_from)

    print("\n训练完成，可以开始翻译")
    return trainer, model



# 推理命令

def cmd_infer(text: str = None):

    model_path = os.path.join(config.save_dir, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"未找到模型文件: {model_path}")

        return

    if text:
        # 单句翻译
        single_translate(model_path, config.tokenizer_path, text)
    else:
        # 交互模式
        interactive_inference(model_path, config.tokenizer_path)


# 测试命令

def cmd_test():
 
    print("\n" + "---" * 20)
    print("  Transformer 教育版 - 模块测试")
    print("---" * 20 + "\n")

    import torch


    print("=" * 50)
    print("  测试 1: 分词器")
    print("=" * 50)
    from tokenizer import BpeTokenizer
    if os.path.exists(config.tokenizer_path):
        tokenizer = BpeTokenizer.from_file(config.tokenizer_path)
        print(f"  词表大小: {len(tokenizer)}")
        zh_ids = tokenizer.encode_zh("你好世界")
        en_ids = tokenizer.encode_en("hello world")
        print(f"  '你好世界' {zh_ids}")
        print(f"  'hello world'  {en_ids}")
        print(f"  解码 '{tokenizer.decode(en_ids)}'")
    else:
        print(f"  未找到 BPE 模型，跳过分词器测试（先跑 train_tokenizer.py）")
        tokenizer = None

    print(f"\n{'='*50}")
    print(f" 模型构建")
    print(f"{'='*50}")
    model = Seq2SeqTransformer(
        vocab_size=1000, d_model=128, n_heads=4,
        n_encoder_layers=2, n_decoder_layers=2,
        d_ff=256, dropout=0.1, max_len=32, pad_idx=0,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数数量: {param_count:,}")


    print(f"\n{'='*50}")
    print(f"  前向传播 ")
    print(f"{'='*50}")

    B, S, T = 2, 8, 6   # batch=2, src_len=8, tgt_len=6
    src = torch.randint(4, 100, (B, S))       # 随机token ID [4,100)
    tgt_in = torch.randint(4, 100, (B, T))
    src_mask = torch.ones(B, S, dtype=torch.bool)
    tgt_mask = torch.ones(B, T, dtype=torch.bool)

    logits = model(src, tgt_in, src_mask, tgt_mask, verbose=True)
    print(f"\n  输出形状: {logits.shape}  (预期: ({B}, {T}, 1000))")


    print(f"\n{'='*50}")
    print(f"   Mask 创建")
    print(f"{'='*50}")
    causal = model.create_causal_mask(5, torch.device("cpu"))
    print(f"  因果 mask (5x5): 0=可见, -inf=屏蔽")
    print(causal.squeeze())


    print(f"\n{'='*50}")
    print(f"   Label Smoothing 损失")
    print(f"{'='*50}")
    from train import LabelSmoothingLoss
    criterion = LabelSmoothingLoss(vocab_size=1000, smoothing=0.1)
    tgt_out = torch.randint(4, 100, (B, T))
    loss = criterion(logits, tgt_out)
    print(f"  Loss (smoothing=0.1): {loss.item():.4f}")

    print(f"\n{'='*50}")
    print(f"  推理")
    print(f"{'='*50}")
    from inference import Translator
    if tokenizer is not None:
        # 推理测试：用 config 的词表大小建小模型，保证和 BPE 词表一致
        infer_model = Seq2SeqTransformer(
            vocab_size=config.vocab_size, d_model=128, n_heads=4,
            n_encoder_layers=2, n_decoder_layers=2,
            d_ff=256, dropout=0.0, max_len=config.max_pos_len, pad_idx=0,
        )
        translator = Translator(infer_model, tokenizer, device="cpu",
                                max_len=config.max_seq_len)
        result = translator.translate("你好")
        print(f" 生成 token: {result['tokens']}")
    else:
        print(f"  未找到 BPE 模型，跳过推理测试")

    print(f"\n{'='*50}")


# 主函数
def main():

    parser = argparse.ArgumentParser( )

    # 子命令
    subparsers = parser.add_subparsers(
        dest="command", help="子命令")

    # train 子命令
    p_train = subparsers.add_parser("train", help="训练")
    p_train.add_argument("--resume", action="store_true",
                         help="从检查点恢复训练")

    # infer 子命令
    p_infer = subparsers.add_parser("infer", help="翻译")
    p_infer.add_argument("text", nargs="?", default=None,
                         help="要翻译的中文文本")

    # test 子命令
    subparsers.add_parser("test", help="测试所有模块")


    args = parser.parse_args()



    if args.command == "train":
        cmd_train(resume=args.resume)
    elif args.command == "infer":
        cmd_infer(text=args.text)
    elif args.command == "test":
        cmd_test()
    else:

        parser.print_help()
        print("  python main.py train")
        print("  python main.py infer")
        print("  python main.py infer")
        print("  python main.py test")


if __name__ == "__main__":
    main()
