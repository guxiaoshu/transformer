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
        【主翻译方法 — 自回归贪心解码（KV Cache 加速）】

        【解码流程】
          1. 中文 → token IDs
          2. 编码器: token IDs → enc_out + cross_kv_caches (一次运行)
          3. 自回归循环 (KV Cache 模式):
             for step in max_len:
               a. 只传入最新 token [1, 1]（不是完整序列！）
               b. 解码器: 新 token 的 Q → 对缓存的 K/V 做注意力
               c. 只需计算新 token 的 K/V 并追加到缓存
               d. 输出头 → logits → 选概率最高的 token
               e. 如果是EOS → 停止
          4. token IDs → 英文文本

        【KV Cache vs 原始方式】
          原始方式：每步重算所有历史 token → O(T³)
          KV Cache：每步只算新 token → O(T²)
          生成 20 个 token：原始方式 ≈ 210 次 K/V 投影，缓存方式 = 20 次

        【参数说明】
          text: 中文输入
          show_weights: 是否收集注意力权重（用于可视化）
          verbose: 是否打印解码过程

        【返回】
          dict: {"translation": str, "tokens": [...], "token_ids": [...], ...}
        """
        self.model.eval()

        # ── 步骤1：编码源语言（一次运行）──
        src_ids = self.tokenizer.encode_zh(text, add_special=True)
        src_ids = src_ids[:self.max_len]

        src_tensor = torch.tensor(
            [src_ids], dtype=torch.long, device=self.device
        )  # (1, S)
        src_mask = torch.ones(
            1, len(src_ids), dtype=torch.bool, device=self.device
        )  # (1, S)

        # 编码器运行一次 → 源语言表示 + 交叉注意力 KV 缓存
        # cross_kv_caches: 每层解码器交叉注意力的 K/V（从 enc_out 预计算）
        # 因为 enc_out 在整个推理过程中不变，这些 K/V 只算一次
        enc_out, cross_kv_caches = self.model.encode_for_inference(
            src_tensor, src_mask, verbose=verbose, return_caches=True
        )

        if verbose:
            src_tokens = [self.tokenizer.itos.get(i, "?") for i in src_ids]
            print(f"\n  源句子 token: {src_tokens}")
            print(f"  编码器输出: {enc_out.shape}")
            print(f"  交叉注意力缓存: {len(cross_kv_caches)} 层")

        # ── 步骤2：初始化 KV Cache ──
        # 每层解码器一个自注意力缓存，初始 k 和 v 都为 None
        # dict 传引用 → 各层的 self_attn.forward() 会原地更新
        self_kv_caches = [
            {"k": None, "v": None}
            for _ in range(len(self.model.decoder_layers))
        ]

        # ── 步骤3：自回归解码（KV Cache 模式）──
        generated_ids = [self.sos_idx]          # 起始符号
        all_attn_weights = [] if show_weights else None

        for step in range(self.max_len - 1):
            # ═══════════════════════════════════════════════════
            # KV Cache 关键：只传入最新 token，不是完整序列！
            # tgt_tensor shape: [1, 1]（vs 原始方式的 [1, past_len+1]）
            # ═══════════════════════════════════════════════════
            new_token = torch.tensor(
                [[generated_ids[-1]]], dtype=torch.long, device=self.device
            )  # (1, 1) — 只有最新 token

            # decode_step 在 KV Cache 模式下：
            #   - past_len=step 用于取正确位置的位置编码
            #   - self_kv_caches 在各层自注意力中被原地更新
            #   - tgt_mask=None（单 token 不需要因果 mask）
            dec_out = self.model.decode_step(
                new_token, enc_out, None, src_mask,  # tgt_mask 不需要
                past_len=step,
                verbose=verbose,
                self_kv_caches=self_kv_caches,
                cross_kv_caches=cross_kv_caches,
            )  # (1, 1, D)

            # 投影到词表 → logits → probabilities
            logits = self.model.output_proj(dec_out)   # (1, 1, V)
            probs = F.softmax(logits, dim=-1)           # 转为概率

            # 选概率最高的token（贪心）
            next_token_id = probs.argmax(dim=-1).item()

            if verbose:
                next_token = self.tokenizer.itos.get(next_token_id, "?")
                prob = probs.max().item()
                cache_len = self_kv_caches[0]["k"].shape[2] if self_kv_caches[0]["k"] is not None else 0
                print(f"    Step {step+1}: 生成 '{next_token}' "
                      f"(id={next_token_id}, p={prob:.3f}, "
                      f"cache_size={cache_len})")

            # 收集注意力权重（如果开启）
            if show_weights:
                weights = self.model.get_attention_weights()
                all_attn_weights.append(weights)

            # EOS → 停止
            if next_token_id == self.eos_idx:
                break

            # 追加到生成序列
            generated_ids.append(next_token_id)

        # ── 步骤4：解码为文本 ──
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

    # ======================================================================
    # 1b. Beam Search 解码
    # ======================================================================

    @torch.no_grad()
    def translate_beam(self, text: str, beam_size: int = 4,
                       show_weights: bool = False,
                       verbose: bool = False) -> dict:
        """
        【Beam Search 束搜索解码 — 在翻译质量和速度之间取平衡】

        【为什么需要 Beam Search】
          贪心解码每一步选概率最高的 token。但局部最优 ≠ 全局最优。
          例如：生成 "the" 之后，概率最高的下一词是 "cat"，
          但 "the black cat" 整体概率可能高于 "the cat" 的后续任何路径。

          Beam Search 同时保留 beam_size 条候选路径，
          每步对所有路径扩展 top-k 个子 token，再从中选出整体最优的 beam_size 条。

        【Beam Search 与 KV Cache】
          此实现不含 KV Cache（每条 beam 需独立缓存，做完整序列重算）。
          因为不同 beam 的 token 序列不同，它们的 self-attention K/V 也不同。
          生产系统中用 PagedAttention 实现 beam 间共享 prompt 部分的 KV 块。
          教育版为了清晰展示算法，使用最简单的完整重算方式。

        【长度惩罚】
          不对 log 概率做长度标准化会导致模型偏向短序列。
          这里使用 length_penalty = len(seq)^0.6 做温和惩罚。

        【算法复杂度】
          每步：beam_size 次 decode_step + beam_size × vocab 次比较
          总：O(beam_size × max_len × T²) ≈ 4 × 32 × (32²) ≈ 131K 次 K/V 投影
          （对比贪心 ≈ 32² ≈ 1K 次，Beam Search 慢约 beam_size × T 倍）

        【参数说明】
          text: 中文输入
          beam_size: 束宽（候选路径数），通常 3~5
          show_weights: 是否返回注意力权重（仅对最佳路径）
          verbose: 是否打印搜索过程

        【返回】
          dict: {
            "translation": str,         # 最佳翻译
            "tokens": List[str],
            "token_ids": List[int],
            "score": float,             # 最佳路径的累积 log 概率
            "num_beams": int,           # 使用的束宽
            "num_candidates": int,      # 探索过的候选路径数
          }
        """
        self.model.eval()

        # ── 步骤1：编码源语言（和贪心一样，只跑一次）──
        src_ids = self.tokenizer.encode_zh(text, add_special=True)
        src_ids = src_ids[:self.max_len]

        src_tensor = torch.tensor(
            [src_ids], dtype=torch.long, device=self.device
        )
        src_mask = torch.ones(
            1, len(src_ids), dtype=torch.bool, device=self.device
        )

        # Beam Search 使用原始 decode_step（无 KV Cache），
        # 因为每条 beam 有不同的 token 序列，各自需要独立的 self-attention 上下文
        enc_out = self.model.encode_for_inference(
            src_tensor, src_mask, verbose=verbose, return_caches=False
        )

        if verbose:
            src_tokens = [self.tokenizer.itos.get(i, "?") for i in src_ids]
            print(f"\n  [Beam Search] 源句子: {src_tokens}")
            print(f"  [Beam Search] beam_size={beam_size}")

        # ── 步骤2：初始化 beams ──
        # 每条 beam 存储 (累积_log_概率, token_id_序列)
        # 使用 log 概率而非原始概率，避免浮点下溢
        beams = [(0.0, [self.sos_idx])]
        completed_beams = []   # 已经遇到 EOS 的完整序列

        for step in range(self.max_len - 1):
            all_candidates = []

            for cum_log_prob, seq in beams:
                # ── 已经完成的 beam 不再扩展 ──
                if seq[-1] == self.eos_idx:
                    completed_beams.append((cum_log_prob, seq))
                    continue

                # ── 对当前 beam 做一次完整前向传播 ──
                tgt_tensor = torch.tensor(
                    [seq], dtype=torch.long, device=self.device
                )  # (1, len(seq))
                tgt_mask_t = torch.ones(
                    1, len(seq), dtype=torch.bool, device=self.device
                )

                dec_out = self.model.decode_step(
                    tgt_tensor, enc_out, tgt_mask_t, src_mask,
                    past_len=len(seq) - 1,
                    # self_kv_caches=None → 使用原始模式（完整重算）
                )  # (1, 1, D)

                # 投影到词表
                logits = self.model.output_proj(dec_out)    # (1, 1, V)
                log_probs = F.log_softmax(logits, dim=-1)   # log 概率
                log_probs = log_probs.squeeze()              # (V,)

                # ── 取当前 beam 的 top-(beam_size × 2) 个候选 ──
                # 为什么 ×2？保留更多候选有助于避免过早剪掉有潜力的路径
                topk_log_probs, topk_ids = log_probs.topk(
                    min(beam_size * 2, log_probs.size(0))
                )

                for i in range(len(topk_ids)):
                    candidate_seq = seq + [topk_ids[i].item()]
                    candidate_score = cum_log_prob + topk_log_probs[i].item()
                    all_candidates.append((candidate_score, candidate_seq))

            # ── 按累积分数排序，保留最好的 beam_size 条 ──
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:beam_size]

            if verbose:
                top_score = beams[0][0] if beams else float('-inf')
                active = sum(1 for b in beams if b[1][-1] != self.eos_idx)
                print(f"    Step {step+1}: {len(all_candidates)} candidates → "
                      f"{len(beams)} beams (active={active}, "
                      f"best_score={top_score:.2f})")

            # ── 所有 active beam 都已完成 → 提前结束 ──
            if all(b[1][-1] == self.eos_idx for b in beams):
                if verbose:
                    print(f"  [Beam Search] 所有 beam 已完成，提前结束")
                break

        # ── 步骤3：从所有候选中选出最佳路径 ──
        # 合并已完成的 beam 和仍在进行中的 beam
        completed_beams.extend(
            [b for b in beams if b[1][-1] == self.eos_idx]
        )

        if not completed_beams:
            # 没有 beam 遇到 EOS（极端情况）→ 使用当前最佳 beam
            completed_beams = beams

        # ── 长度惩罚：避免模型偏向过短的序列 ──
        # score = log_prob / len(seq)^α
        # α=0.6 是经验值（Google NMT 论文），在"太短"和"太长"之间取平衡
        # 不加惩罚 → 模型倾向 2-3 词的短句
        # α 太大 → 模型倾向超长句
        best_score, best_seq = max(
            completed_beams,
            key=lambda x: x[0] / (len(x[1]) ** 0.6)
        )

        translation = self.tokenizer.decode(best_seq, skip_special=True)

        if verbose:
            print(f"  [Beam Search] 最佳路径: score={best_score:.2f}, "
                  f"len={len(best_seq)}, {len(completed_beams)} 条候选")
            print(f"  [Beam Search] 翻译: '{translation}'")

        return {
            "translation": translation,
            "tokens": [self.tokenizer.itos.get(i, "?") for i in best_seq],
            "token_ids": best_seq,
            "score": best_score,
            "num_beams": beam_size,
            "num_candidates": len(completed_beams),
        }


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
    print("    输入中文 → 获得英文翻译（贪心解码 + KV Cache）")
    print("    输入中文 --beam  → Beam Search 解码（质量更高，速度较慢）")
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
        use_beam = "--beam" in user_input
        show_weights = "--weight" in user_input
        verbose = "--verbose" in user_input
        user_input = user_input.replace("--beam", "").replace("--weight", "").replace("--verbose", "").strip()

        if not user_input:
            continue

        # ── 翻译 ──
        start = time.time()
        if use_beam:
            # Beam Search 模式（无 KV Cache，但质量更高）
            result = translator.translate_beam(
                user_input,
                beam_size=config.beam_size,
                show_weights=show_weights,
                verbose=verbose,
            )
            elapsed = time.time() - start
            print(f"  [EN] 英文 > {result['translation']}")
            print(f"  (耗时: {elapsed:.2f}s, "
                  f"beam={result['num_beams']}, "
                  f"score={result['score']:.2f})")
        else:
            # 贪心解码 + KV Cache
            result = translator.translate(
                user_input,
                show_weights=show_weights,
                verbose=verbose,
            )
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
                     show_weights: bool = False, verbose: bool = False,
                     use_beam: bool = False):
    """
    【单次翻译 — 非交互模式】

    用于命令行直接调用:
      python main.py infer "你好世界" --weight
      python main.py infer "你好世界" --beam

    【参数说明】
      model_path: .pt 检查点路径
      tokenizer_path: tokenizer.json 路径
      text: 要翻译的中文
      show_weights: 显示注意力
      verbose: 打印解码过程
      use_beam: 使用 Beam Search（否则用贪心 + KV Cache）
    """
    translator = Translator.from_checkpoint(model_path, tokenizer_path)

    if use_beam:
        # Beam Search 解码（质量优先）
        result = translator.translate_beam(
            text,
            beam_size=config.beam_size,
            show_weights=show_weights,
            verbose=verbose,
        )
    else:
        # 贪心解码 + KV Cache（速度优先）
        result = translator.translate(
            text,
            show_weights=show_weights,
            verbose=verbose,
        )

    print(f"\n  中文: {text}")
    print(f"  英文: {result['translation']}")
    if use_beam:
        print(f"  (Beam Search: beam={result['num_beams']}, "
              f"score={result['score']:.2f})")

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
        show_b = "--beam" in text
        text = text.replace("--weight", "").replace("--beam", "").strip()
        single_translate(
            os.path.join(config.save_dir, "best_model.pt"),
            config.tokenizer_path,
            text,
            show_weights=show_w,
            verbose=True,
            use_beam=show_b,
        )
    else:
        interactive_inference()
