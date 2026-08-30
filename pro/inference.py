import os
import time
import torch
import torch.nn.functional as F

from config import config
from model import Seq2SeqTransformer
from tokenizer import BpeTokenizer



class Translator:


    def __init__(self, model: Seq2SeqTransformer, tokenizer: BpeTokenizer,
                 device: str = "cuda", max_len: int = 64):

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

        # 特殊token ID
        self.pad_idx = tokenizer.PAD_IDX   # 0
        self.sos_idx = tokenizer.SOS_IDX   # 1
        self.eos_idx = tokenizer.EOS_IDX   # 2

        # 推理模式
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str = None):

        if tokenizer_path is None:
            tokenizer_path = config.tokenizer_path

 
        tokenizer = BpeTokenizer.from_file(tokenizer_path)

 
        model = Seq2SeqTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_ff=config.d_ff,
            dropout=0.0,               # 推理时不需要dropout
            max_len=config.max_pos_len,
            pad_idx=0,
        )

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=config.device)


        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"从 epoch "
                  f"{checkpoint.get('epoch', '?')} 检查点加载")
        else:
            model.load_state_dict(checkpoint)
            print(f"从权重文件加载")

        return cls(model, tokenizer, device=config.device,
                   max_len=config.max_seq_len)

    # 自回归贪心解码

    @torch.no_grad()   # 推理不需要梯度，节省显存
    def translate(self, text: str, verbose: bool = False) -> dict:

        self.model.eval()

        # 编码源语言
        src_ids = self.tokenizer.encode_zh(text, add_special=True)
        src_ids = src_ids[:self.max_len]

        src_tensor = torch.tensor(
            [src_ids], dtype=torch.long, device=self.device
        )  # (1, S)
        src_mask = torch.ones(
            1, len(src_ids), dtype=torch.bool, device=self.device
        )  # (1, S)

        # 编码器运行一次
        # cross_kv_caches: 每层解码器交叉注意力的 KV（从 enc_out 预计算）
        # 因为 enc_out 在整个推理过程中不变，这些  KV 只算一次
        enc_out, cross_kv_caches = self.model.encode_for_inference(
            src_tensor, src_mask, verbose=verbose, return_caches=True
        )

        if verbose:
            src_tokens = [self.tokenizer.id_to_piece(i) for i in src_ids]
            print(f"\n  源句子 token {src_tokens}")
            print(f"  编码器输出 {enc_out.shape}")
            print(f"  交叉注意力缓存 {len(cross_kv_caches)} 层")

        # 初始化 KV Cache 
        # 每层解码器一个自注意力缓存，初始 k 和 v 都为 None

        self_kv_caches = [
            {"k": None, "v": None}
            for _ in range(len(self.model.decoder_layers))
        ]

        # ：自回归解码（KV Cache）
        generated_ids = [self.sos_idx]          # 起始符号

        for step in range(self.max_len - 1):

            # 只传入最新 token，不是完整序列！
            # tgt_tensor shape: [1, 1]，， 原始方式 [1, past_len+1]）
 
            new_token = torch.tensor(
                [[generated_ids[-1]]], dtype=torch.long, device=self.device
            )  # (1, 1) — 只有最新 token

            dec_out = self.model.decode_step(
                new_token, enc_out, None, src_mask,  # tgt_mask 不需要
                past_len=step,
                verbose=verbose,
                self_kv_caches=self_kv_caches,
                cross_kv_caches=cross_kv_caches,
            )  # (1, 1, D)

            logits = self.model.output_proj(dec_out)   # (1, 1, V)
            probs = F.softmax(logits, dim=-1)           # 转为概率

            # 选概率最高的token
            next_token_id = probs.argmax(dim=-1).item()


            # EOS
            if next_token_id == self.eos_idx:
                break

            # 追加到生成序列
            generated_ids.append(next_token_id)

        # 解码为文本
        translation = self.tokenizer.decode(generated_ids, skip_special=True)

        result = {
            "translation": translation,
            "tokens": [self.tokenizer.id_to_piece(i)
                       for i in generated_ids],
            "token_ids": generated_ids,
        }

        return result

# 交互式推理 

def interactive_inference(model_path: str = None, tokenizer_path: str = None):

    if model_path is None:
        model_path = os.path.join(config.save_dir, "best_model.pt")
    if tokenizer_path is None:
        tokenizer_path = config.tokenizer_path

    print("\n" + "=" * 60)

    print("=" * 60)
    print(f"  模型 {model_path}")
    print(f"  词表 {tokenizer_path}")

    # 加载翻译器
    translator = Translator.from_checkpoint(model_path, tokenizer_path)
    param_count = sum(p.numel() for p in translator.model.parameters())
    print(f"  模型参数 {param_count:,}")
    print(f"  设备 {translator.device}")
    print()
    print("  使用方法:")
    print("    输入中文获得英文翻译")
    print("    输入 exit → 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("  中文 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("  再见")
            break


        verbose = "--verbose" in user_input
        user_input = user_input.replace("--verbose", "").strip()

        if not user_input:
            continue

        # 翻译 
        start = time.time()
        # 贪心解码 + KV Cache
        result = translator.translate(
            user_input,
            verbose=verbose,
        )
        elapsed = time.time() - start
        print(f"  英文 > {result['translation']}")
        print(f"  (耗时: {elapsed:.2f}s, "
              f"生成长度: {len(result['tokens'])} tokens)")


        print()


def single_translate(model_path: str, tokenizer_path: str, text: str,
                     verbose: bool = False):

    translator = Translator.from_checkpoint(model_path, tokenizer_path)

    result = translator.translate(
        text,
        verbose=verbose,
    )



    return result


# 命令行直接调用
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        single_translate(
            os.path.join(config.save_dir, "best_model.pt"),
            config.tokenizer_path,
            text,
            verbose=True,
        )
    else:
        interactive_inference()
