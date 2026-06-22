"""
================================================================================
Transformer 教育版 — 推理与注意力可视化模块 (Inference)
================================================================================

【本模块在Pipeline中的位置】
  用户输入中文 → [inference.py] → 英文翻译 + (可选)注意力权重热力图

【解码策略】
  贪心解码 (Greedy Decoding): 每一步选概率最高的token
    - 优点: 速度最快，实现简单
    - 缺点: 没有回溯，早期错误无法修正
    - 为什么教育版用这个: 简单直观，方便展示注意力

【自回归生成流程】
  1. 编码器一次运行: "你好" → enc_out
  2. 解码器循环:
     Step 1: 输入 [SOS]        → 预测 "hello"
     Step 2: 输入 [SOS, hello] → 预测 "world"
     Step 3: 输入 [SOS, hello, world] → 预测 EOS → 停止

【注意力可视化】
  启动方式: 翻译时加 --weight 参数
  显示内容:
    - 交叉注意力热力图 (解码器→编码器)
    - 每个生成步骤，模型在源句子上"看"了哪些位置
    - USE ASCII 终端字符表示注意力强度

================================================================================
"""

import os
import math
import time
import torch
import torch.nn.functional as F

from config import config
from model import Seq2SeqTransformer
from tokenizer import SimpleTokenizer


# ==============================================================================
# 1. 翻译器
# ==============================================================================

class Translator:
    """
    翻译器 — 加载训练好的模型，执行中文→英文翻译。

    【核心方法】
      translate(text, show_weights=False):
        自回归贪心解码，可选注意力权重收集

    【模型加载方式】
      from_checkpoint(): 从 .pt 文件加载权重
    """

    def __init__(self, model: Seq2SeqTransformer, tokenizer: SimpleTokenizer,
                 device: str = "cuda", max_len: int = 64):
        """
        【参数说明】
          model: 已加载权重的 Seq2SeqTransformer
          tokenizer: 已加载词表的 SimpleTokenizer
          device: "cuda" 或 "cpu"
          max_len: 最大生成长度（防止无限循环）
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

        # 特殊token ID（从tokenizer获取，保持一致性）
        self.pad_idx = tokenizer.PAD_IDX   # 0
        self.sos_idx = tokenizer.SOS_IDX   # 1
        self.eos_idx = tokenizer.EOS_IDX   # 2

        # 切换到推理模式（关闭 Dropout）
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, tokenizer_path: str = None):
        """
        【从文件加载翻译器】

        【加载流程】
          1. 加载分词器词表
          2. 创建模型结构（随机权重）
          3. 加载训练好的权重
          4. 组装翻译器

        【为什么需要分"创建结构"和"加载权重"两步】
          PyTorch 的 save/load 机制要求模型结构完全一致。
          如果修改了模型结构，旧检查点无法加载。
          这里确保创建的和保存时相同的结构。
        """
        if tokenizer_path is None:
            tokenizer_path = config.tokenizer_path

        # ── 加载分词器 ──
        tokenizer = SimpleTokenizer.load(tokenizer_path)

        # ── 创建模型结构 ──
        # dropout=0.0: 推理时不需要dropout（且model.eval()也会禁用它）
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

        # ── 加载权重 ──
        checkpoint = torch.load(checkpoint_path, map_location=config.device)

        # 兼容两种保存格式：
        #   Trainer.save(): {"model_state_dict": ..., "optimizer_state_dict": ...}
        #   纯权重文件:     直接是 state_dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Translator] 从 epoch "
                  f"{checkpoint.get('epoch', '?')} 检查点加载")
        else:
            model.load_state_dict(checkpoint)
            print(f"[Translator] 从权重文件加载")

        return cls(model, tokenizer, device=config.device,
                   max_len=config.max_seq_len)

    # ======================================================================
    # 1a. 自回归贪心解码
    # ======================================================================

    @torch.no_grad()   # 推理不需要梯度，节省显存
    def translate(self, text: str, show_weights: bool = False,
                  verbose: bool = False) -> dict:
        """
        【主翻译方法 — 自回归贪心解码】

        【解码流程】
          1. 中文 → token IDs
          2. 编码器: token IDs → enc_out (一次运行)
          3. 自回归循环:
             for step in max_len:
               a. 解码器: 当前token序列 → 下一步的特征
               b. 输出头: 特征 → logits → softmax → 概率
               c. argmax: 选概率最高的token
               d. 如果是EOS → 停止
               e. 加入生成序列，继续循环
          4. token IDs → 英文文本

        【Teacher Forcing vs 自回归】
          训练 (Teacher Forcing): 每一步输入真实的前一个token
          推理 (自回归): 每一步输入模型自己生成的前一个token
          如果模型早期生成错了，后面会越来越偏 → 错误累积

        【参数说明】
          text: 中文输入 "你好"
          show_weights: 是否收集注意力权重（用于可视化）
          verbose: 是否打印解码过程

        【返回】
          dict: {
            "translation": str,        # 翻译结果文本
            "tokens": List[str],       # 每个生成token的文本
            "token_ids": List[int],    # 每个生成token的ID
            "attention_weights": ...,  # (如果show_weights=True)
            "src_tokens": ...,         # (如果show_weights=True)
          }
        """
        self.model.eval()

        # ── 步骤1：编码源语言 ──
        # 中文 → token IDs → tensor
        src_ids = self.tokenizer.encode_zh(text, add_special=True)
        src_ids = src_ids[:self.max_len]

        src_tensor = torch.tensor(
            [src_ids], dtype=torch.long, device=self.device
        )  # (1, S)
        src_mask = torch.ones(
            1, len(src_ids), dtype=torch.bool, device=self.device
        )  # (1, S)

        # 编码器运行一次，得到源语言表示
        enc_out = self.model.encode_for_inference(
            src_tensor, src_mask, verbose=verbose
        )

        if verbose:
            src_tokens = [self.tokenizer.itos.get(i, "?") for i in src_ids]
            print(f"\n  源句子 token: {src_tokens}")
            print(f"  编码器输出: {enc_out.shape}")

        # ── 步骤2：自回归解码 ──
        # 从 SOS 开始
        generated_ids = [self.sos_idx]          # [1]
        all_attn_weights = [] if show_weights else None

        for step in range(self.max_len - 1):
            # 构造当前输入（包含所有历史token）
            tgt_tensor = torch.tensor(
                [generated_ids], dtype=torch.long, device=self.device
            )  # (1, T)
            tgt_len = len(generated_ids)
            tgt_mask_tensor = torch.ones(
                1, tgt_len, dtype=torch.bool, device=self.device
            )

            # 单步解码：输入已生成序列 → 得到最后一个位置的特征
            dec_out = self.model.decode_step(
                tgt_tensor, enc_out, tgt_mask_tensor, src_mask, past_len=step,
                verbose=verbose,
            )  # (1, 1, D)

            # 投影到词表 → logits → probabilities
            logits = self.model.output_proj(dec_out)   # (1, 1, V)
            probs = F.softmax(logits, dim=-1)           # 转为概率

            # 选概率最高的token（贪心）
            next_token_id = probs.argmax(dim=-1).item()

            if verbose:
                next_token = self.tokenizer.itos.get(next_token_id, "?")
                prob = probs.max().item()
                print(f"    Step {step+1}: 生成 '{next_token}' "
                      f"(id={next_token_id}, p={prob:.3f})")

            # 收集注意力权重（如果开启）
            if show_weights:
                # get_attention_weights 返回每层的自注意力和交叉注意力
                weights = self.model.get_attention_weights()
                all_attn_weights.append(weights)

            # EOS → 停止
            if next_token_id == self.eos_idx:
                break

            # 追加到生成序列
            generated_ids.append(next_token_id)

        # ── 步骤3：解码为文本 ──
        translation = self.tokenizer.decode(generated_ids, skip_special=True)

        result = {
            "translation": translation,
            "tokens": [self.tokenizer.itos.get(i, "?")
                       for i in generated_ids],
            "token_ids": generated_ids,
        }

        if show_weights:
            result["attention_weights"] = all_attn_weights
            result["src_tokens"] = [self.tokenizer.itos.get(i, "?")
                                     for i in src_ids]

        return result


# ==============================================================================
# 2. 注意力可视化
# ==============================================================================

def print_attention_weights(result: dict, layer_idx: int = -1):
    """
    【终端注意力热力图 — ASCII 可视化】

    显示解码器→编码器的交叉注意力。
    每一行 = 一个解码步骤（生成的英文token）
    每一列 = 一个源语言位置（中文token）
    字符密度 = 注意力强度

    【为什么看交叉注意力】
      交叉注意力是模型在"翻译"时的视野：
      - 生成"hello"时，模型在源句子上看了哪个字？
      - 生成"world"时，注意力是否移动到了"世界"？
      这是理解模型行为的直接窗口。

    【参数说明】
      result: translate() 的返回结果
      layer_idx: 显示第几层 (-1=最后一层, 0=第一层)
    """
    if "attention_weights" not in result or not result["attention_weights"]:
        print("  (无注意力权重数据)")
        return

    src_tokens = result["src_tokens"]
    tgt_tokens = result["tokens"]
    all_weights = result["attention_weights"]
    num_steps = len(all_weights)
    num_layers = len(all_weights[0]) if num_steps > 0 else 0

    if num_steps == 0 or layer_idx >= num_layers:
        return

    # 取指定层
    l = layer_idx

    # ── 重建所有 step 的交叉注意力 ──
    num_src = len(src_tokens)
    attn_matrix = torch.zeros(min(num_steps, 20), num_src)

    for step_idx in range(min(num_steps, 20)):
        # all_weights[step][layer]["cross"]: (1, H, 1, S)
        cross = all_weights[step_idx][l]["cross"]
        # 对所有头取平均 → (1, 1, S) → (S,)
        cross_avg = cross[0].mean(dim=0)[0].cpu()
        s_len = min(cross_avg.shape[0], num_src)
        attn_matrix[step_idx, :s_len] = cross_avg[:s_len]

    # ── 打印 ──
    # Unicode block字符：按注意力强度 0→1 映射到不同密度
    blocks = [" ", "░", "▒", "▓", "█"]

    print("\n" + "=" * 80)
    print(f"  注意力权重可视化 (解码器 → 编码器, Layer {l})")
    print(f"  行=解码步骤, 列=源token | 越密=越关注")
    print("=" * 80)

    # 列标题
    header = f"{'Step':>4s} | {'Token':<12s}"
    max_cols = min(num_src, 25)
    for i in range(max_cols):
        tok = src_tokens[i] if i < len(src_tokens) else "?"
        header += f"{tok[:3]:>3s}"
    print(header)
    print("-" * (20 + 3 * max_cols))

    # 每行 = 一个生成步骤
    for step in range(min(num_steps, 20)):
        tgt_tok = tgt_tokens[step] if step < len(tgt_tokens) else "?"
        line = f"{step:4d} | {tgt_tok[:10]:<12s}"

        for s in range(max_cols):
            if s < num_src:
                val = attn_matrix[step, s].item()
                idx = min(int(val * 5), 4)   # 0→0.0, 1→0.2, ..., 4→0.8+
                line += f" {blocks[idx]}"
            else:
                line += "  "

        print(line)

    # 图例
    print(f"\n  图例: "
          f"{' '.join(f'{b}={i/4:.1f}' for i, b in enumerate(blocks))}")
    print("=" * 80)


def print_all_layer_weights(result: dict):
    """
    【所有层的注意力摘要】
    显示每层交叉注意力中模型最关注的几个源token。
    不同层关注不同信息：浅层更局部，深层更全局。
    """
    if "attention_weights" not in result or not result["attention_weights"]:
        return

    src_tokens = result["src_tokens"]
    all_weights = result["attention_weights"]
    num_steps = len(all_weights)
    num_layers = len(all_weights[0]) if num_steps > 0 else 0

    if num_steps == 0:
        return

    print("\n" + "-" * 60)
    print("  每层交叉注意力 Top-3 关注的源 token (最后生成步骤)")
    print("-" * 60)

    for l in range(num_layers):
        # 取最后一步、第l层、交叉注意力
        cross = all_weights[-1][l]["cross"]     # (1, H, 1, S)
        cross_avg = cross[0].mean(dim=0)[0]     # (S,) 对所有头取平均
        top3 = cross_avg.topk(min(3, len(src_tokens)))
        top_str = ", ".join(
            f"{src_tokens[i]}({cross_avg[i].item():.2f})"
            for i in top3.indices
        )
        print(f"  Layer {l}: {top_str}")


# ==============================================================================
# 3. 交互式推理
# ==============================================================================

def interactive_inference(model_path: str = None, tokenizer_path: str = None):
    """
    【交互式推理 — 终端中的翻译对话】

    类似聊天界面：
      用户输入中文
      模型输出英文
      可选 --weight 显示注意力
      可选 --verbose 显示解码过程

    【启动方式】
      python main.py infer
    """
    if model_path is None:
        model_path = os.path.join(config.save_dir, "best_model.pt")
    if tokenizer_path is None:
        tokenizer_path = config.tokenizer_path

    print("\n" + "=" * 60)
    print("  Transformer 中→英 翻译推理")
    print("=" * 60)
    print(f"  模型: {model_path}")
    print(f"  词表: {tokenizer_path}")

    # 加载翻译器
    translator = Translator.from_checkpoint(model_path, tokenizer_path)
    param_count = sum(p.numel() for p in translator.model.parameters())
    print(f"  模型参数: {param_count:,}")
    print(f"  设备: {translator.device}")
    print()
    print("  使用方法:")
    print("    输入中文 → 获得英文翻译")
    print("    输入中文 --weight → 翻译 + 注意力权重")
    print("    输入中文 --verbose → 翻译 + 解码过程")
    print("    输入 exit → 退出")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("  [ZH] 中文 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("  再见!")
            break

        # ── 解析参数 ──
        show_weights = "--weight" in user_input
        verbose = "--verbose" in user_input
        user_input = user_input.replace("--weight", "").replace("--verbose", "").strip()

        if not user_input:
            continue

        # ── 翻译 ──
        start = time.time()
        result = translator.translate(user_input,
                                       show_weights=show_weights,
                                       verbose=verbose)
        elapsed = time.time() - start

        print(f"  [EN] 英文 > {result['translation']}")
        print(f"  (耗时: {elapsed:.2f}s, "
              f"生成长度: {len(result['tokens'])} tokens)")

        if verbose:
            print(f"  Token ID 序列: {result['token_ids']}")

        # ── 注意力可视化 ──
        if show_weights and "attention_weights" in result:
            n_layers = len(result["attention_weights"][0]) \
                if result["attention_weights"] else 0
            print(f"\n  共 {n_layers} 个解码器层, "
                  f"{len(result['attention_weights'])} 个生成步骤")

            # 打印最后一层的交叉注意力热力图
            print_attention_weights(result, layer_idx=-1)
            # 打印所有层的摘要
            print_all_layer_weights(result)

        print()


def single_translate(model_path: str, tokenizer_path: str, text: str,
                     show_weights: bool = False, verbose: bool = False):
    """
    【单次翻译 — 非交互模式】

    用于命令行直接调用:
      python inference.py "你好世界" --weight

    【参数说明】
      model_path: .pt 检查点路径
      tokenizer_path: tokenizer.json 路径
      text: 要翻译的中文
      show_weights: 显示注意力
      verbose: 打印解码过程
    """
    translator = Translator.from_checkpoint(model_path, tokenizer_path)
    result = translator.translate(text,
                                   show_weights=show_weights,
                                   verbose=verbose)

    print(f"\n  中文: {text}")
    print(f"  英文: {result['translation']}")

    if show_weights and "attention_weights" in result:
        print_attention_weights(result, layer_idx=-1)
        print_all_layer_weights(result)

    return result


# 命令行直接调用
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        show_w = "--weight" in text
        text = text.replace("--weight", "").strip()
        single_translate(
            os.path.join(config.save_dir, "best_model.pt"),
            config.tokenizer_path,
            text,
            show_weights=show_w,
            verbose=True,
        )
    else:
        interactive_inference()
