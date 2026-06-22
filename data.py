"""
================================================================================
Transformer 教育版 — 数据预处理模块 (Data Pipeline)
================================================================================

【本模块在Pipeline中的位置】
  原始Arrow文件 → [data.py] → DataLoader (可迭代的批量张量)

【数据处理全流程】
  1. 加载:   Arrow IPC → Python列表 (100万条 → 取5万条)
  2. 清洗:   去重 → 过滤长度异常 → 去空白
  3. 分词:   中文逐字 → 英文逐词 → 构建联合词表
  4. 编码:   文本 → token ID序列 (每条: {src, tgt_input, tgt_output})
  5. 批处理: 不等长序列 → padding + mask → 整齐的矩阵

【Teacher Forcing 数据格式】
  假设目标句子 = "hello world":
    tgt_ids     = [SOS, hello, world, EOS] = [1, 18, 20, 2]
    tgt_input   = [SOS, hello, world]      = [1, 18, 20]  (去掉EOS)
    tgt_output  = [hello, world, EOS]      = [18, 20, 2]  (去掉SOS)

  模型看到 tgt_input，预测 tgt_output。
  位置0: 看到[SOS]  → 预测 "hello"
  位置1: 看到[SOS, hello] → 预测 "world"
  位置2: 看到[SOS, hello, world] → 预测 EOS

  这就是 Teacher Forcing：每一步告诉模型"正确的前缀"。
================================================================================
"""

import os
import time
import re
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.ipc as ipc   # Apache Arrow IPC: 读取.arrow文件

from config import config
from tokenizer import SimpleTokenizer


# ==============================================================================
# 1. 加载原始数据
# ==============================================================================

def load_opus100(data_dir: str, src_lang: str = "zh", tgt_lang: str = "en",
                 max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    【从 Arrow 文件加载 OPUS-100 平行语料】

    【Arrow 格式简介】
      Apache Arrow 是一种列式存储格式。
      OPUS-100 的 Arrow 文件每个row包含一个 struct:
        {"translation": {"en": "Hello", "zh": "你好"}}
      我们用 pyarrow.ipc.open_stream() 逐批读取。

    【语言方向交换】
      Arrow 中存储的是 en→zh（英语→中文）。
      但我们要训练 zh→en，所以加载时交换 src/tgt。

    【加载策略】
      用 ipc.open_stream 逐批读取（不是一次性加载全部）。
      每批大约1000条左右（取决于Arrow的分块大小）。
      设置 max_samples 限制后提前停止，节省内存和时间。

    【参数说明】
      data_dir: Arrow文件根目录
      src_lang: "zh" — 源语言
      tgt_lang: "en" — 目标语言
      max_samples: 最多读多少条 (None=全部, 教育版=50000)

    【返回】
      (src_texts, tgt_texts): 两个长度相等的字符串列表
    """
    # Arrow文件路径: {data_dir}/en-zh/0.0.0/{hash}/opus-100-train.arrow
    arrow_dir = os.path.join(data_dir, "en-zh", "0.0.0")

    # HuggingFace datasets 用 hash 命名子目录（内容寻址）
    subdirs = [d for d in os.listdir(arrow_dir)
               if os.path.isdir(os.path.join(arrow_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"未找到 Arrow 文件目录: {arrow_dir}")

    arrow_dir = os.path.join(arrow_dir, subdirs[0])
    train_file = os.path.join(arrow_dir, "opus-100-train.arrow")

    print(f"[Data] 从 {train_file} 加载数据...")

    src_texts = []
    tgt_texts = []

    # ipc.open_stream: 打开Arrow流式文件
    with ipc.open_stream(train_file) as reader:
        # reader 是可迭代的，每次返回一个 RecordBatch
        for batch_idx, batch in enumerate(reader):
            # 取 translation 列 → Python list of dicts
            # 例如 [{"en": "...", "zh": "..."}, ...]
            translations = batch.column("translation").to_pylist()

            for row in translations:
                # 根据方向交换
                if src_lang == "zh":
                    src_texts.append(row["zh"])
                    tgt_texts.append(row["en"])
                else:
                    src_texts.append(row["en"])
                    tgt_texts.append(row["zh"])

                # 达到上限 → 停止
                if max_samples and len(src_texts) >= max_samples:
                    break

            if max_samples and len(src_texts) >= max_samples:
                break

    print(f"[Data] 加载完成: {len(src_texts)} 个平行句对")
    return src_texts, tgt_texts


# ==============================================================================
# 2. 数据清洗
# ==============================================================================

def clean_text(text: str) -> str:
    """单条文本清洗：多余空白→单个空格"""
    return re.sub(r"\s+", " ", text).strip()


def filter_data(src_texts: List[str], tgt_texts: List[str],
                min_len: int = 2, max_len: int = 64) -> Tuple[List[str], List[str]]:
    """
    【数据清洗与过滤】

    【过滤条件】
      1. 空句子：清洗后为空 → 丢弃
      2. 长度比异常：源/目标长度比 > 10 → 丢弃
         例如 "Yes" ↔ "联合国气候变化框架公约缔约方会议" 这种极端不对等
      3. 重复句对：完全相同的 (src, tgt) 组合 → 丢弃

    【为什么需要这些过滤】
      低质量数据 = 训练噪音 = 模型学不到有用规律
      教育版数据量有限（5万条），每个样本都要有"营养"

    【参数说明】
      src_texts: 源语言文本列表
      tgt_texts: 目标语言文本列表
      min_len: 最短字符数
      max_len: 最长字符数（字符级，token后会截断到64）

    【返回】
      过滤后的 (src_texts, tgt_texts)
    """
    print(f"[Data] 过滤前: {len(src_texts)} 句对")

    seen = set()               # 去重用
    clean_src, clean_tgt = [], []

    for src, tgt in zip(src_texts, tgt_texts):
        # ── 清洗空白 ──
        src = clean_text(src)
        tgt = clean_text(tgt)

        # ── 过滤空句子 ──
        if not src or not tgt:
            continue

        # ── 过滤长度比异常 ──
        # 字符级检查（不是token级），放宽到max_len×5
        if len(src) < min_len or len(tgt) < min_len:
            continue
        if len(src) > max_len * 5 or len(tgt) > max_len * 5:
            continue

        # 长度比 > 10:1 → 可能是错误对齐
        ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
        if ratio > 10:
            continue

        # ── 去重 ──
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)

        clean_src.append(src)
        clean_tgt.append(tgt)

    removed = len(src_texts) - len(clean_src)
    print(f"[Data] 过滤后: {len(clean_src)} 句对 (过滤了 {removed})")
    return clean_src, clean_tgt


# ==============================================================================
# 3. 翻译数据集类
# ==============================================================================

class TranslationDataset(Dataset):
    """
    翻译数据集 — 把文本编码为张量，支持按索引取数据。

    【存储格式】
      self.data[i] = {
        "src":       [1, id(你), id(好), 2],        ← 含SOS/EOS
        "tgt_input":  [1, id(hello)],                ← 含SOS, 去掉EOS
        "tgt_output": [id(hello), 2],                ← 去掉SOS, 含EOS
      }

    【为什么预编码】
      如果在 __getitem__ 中实时编码，每条数据都要重新分词+查字典。
      预编码后在内存中存储整数列表，访问数据 = 直接读列表。
      代价是内存消耗（5万条 × 平均长度 = ~几MB，完全可以接受）。
    """

    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 tokenizer: SimpleTokenizer, max_len: int):
        """
        【初始化：逐条编码所有数据】

        【参数说明】
          src_texts: 源语言文本列表
          tgt_texts: 目标语言文本列表
          tokenizer: 已构建词表的 SimpleTokenizer
          max_len: 序列截断长度
        """
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        print(f"[Dataset] 编码 {len(src_texts)} 条数据 (max_len={max_len})...")
        start = time.time()

        for src, tgt in zip(src_texts, tgt_texts):
            # ── 编码源语言（含SOS/EOS） ──
            src_ids = tokenizer.encode_zh(src, add_special=True)
            src_ids = src_ids[:max_len]   # 截断

            # ── 编码目标语言 ──
            tgt_ids = tokenizer.encode_en(tgt, add_special=True)
            tgt_ids = tgt_ids[:max_len]

            # 过滤编码后过短的（至少需要SOS+1个token）
            if len(src_ids) < 2 or len(tgt_ids) < 2:
                continue

            # ── Teacher Forcing 拆分 ──
            tgt_input = tgt_ids[:-1]    # 去掉最后的EOS
            tgt_output = tgt_ids[1:]    # 去掉最前的SOS

            self.data.append({
                "src": src_ids,
                "tgt_input": tgt_input,
                "tgt_output": tgt_output,
            })

        print(f"[Dataset] 编码完成: {len(self.data)} 条, "
              f"耗时 {time.time() - start:.1f}s")

    def __len__(self):
        """PyTorch DataLoader 需要知道数据集大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """PyTorch DataLoader 用此方法逐条取数据"""
        return self.data[idx]


# ==============================================================================
# 4. 批处理函数 (collate_fn)
# ==============================================================================

def collate_fn(batch: List[dict], pad_idx: int = 0, max_len: int = 64):
    """
    【批处理函数 — 把不等长序列填充为整齐矩阵】

    DataLoader 调用此函数把 batch_size 条数据合并为一个 batch。

    【为什么需要collate_fn】
      DataLoader 默认行为是 stack 所有样本，但不等长序列无法直接stack。
      collate_fn 负责"填充→对齐"这最后一步。

    【填充策略】
      - 找到 batch 内最长的序列长度（不超过 max_len）
      - 创建"全0"矩阵（0=PAD_IDX）
      - 逐条复制有效token到对应行
      - 创建 mask 标记哪些位置是有效token

    【输入】
      一个list，包含B个dict:
        [{"src":[1,7,4,2], "tgt_input":[1,18], "tgt_output":[18,2]}, ...]

    【输出】5个张量
      src:        (B, max_src_len)  source token IDs
      src_mask:   (B, max_src_len)  True=有效位置
      tgt_input:  (B, max_tgt_len)  decoder输入
      tgt_mask:   (B, max_tgt_len)  True=有效位置
      tgt_output: (B, max_tgt_len)  decoder目标（和tgt_input一一对应）
    """
    B = len(batch)

    # 找batch内最大长度
    max_src = min(max(len(d["src"]) for d in batch), max_len)
    max_tgt = min(max(len(d["tgt_input"]) for d in batch), max_len)

    # ── 创建全零矩阵 ──
    # torch.full: 填充指定值(dtype=long → 整数)
    src = torch.full((B, max_src), fill_value=pad_idx, dtype=torch.long)
    src_mask = torch.zeros((B, max_src), dtype=torch.bool)

    tgt_input = torch.full((B, max_tgt), fill_value=pad_idx, dtype=torch.long)
    tgt_mask = torch.zeros((B, max_tgt), dtype=torch.bool)
    tgt_output = torch.full((B, max_tgt), fill_value=pad_idx, dtype=torch.long)

    # ── 逐条填入 ──
    for i, d in enumerate(batch):
        # Source
        s_len = min(len(d["src"]), max_src)
        src[i, :s_len] = torch.tensor(d["src"][:s_len], dtype=torch.long)
        src_mask[i, :s_len] = True

        # Target input
        t_len = min(len(d["tgt_input"]), max_tgt)
        tgt_input[i, :t_len] = torch.tensor(d["tgt_input"][:t_len], dtype=torch.long)
        tgt_mask[i, :t_len] = True

        # Target output (长度可能和input差1，这里取相同截断)
        o_len = min(len(d["tgt_output"]), max_tgt)
        tgt_output[i, :o_len] = torch.tensor(d["tgt_output"][:o_len], dtype=torch.long)

    return src, src_mask, tgt_input, tgt_mask, tgt_output


# ==============================================================================
# 5. 创建 DataLoader
# ==============================================================================

def create_dataloaders(src_texts: List[str], tgt_texts: List[str],
                       tokenizer: SimpleTokenizer,
                       batch_size: int = 64, max_len: int = 64,
                       val_ratio: float = 0.02):
    """
    【创建训练/验证 DataLoader】

    【划分策略】
      从数据尾部切 val_ratio(2%) 作为验证集
      教育版: 46377 × 2% ≈ 927条验证

    【DataLoader 参数说明】
      - shuffle=True: 每个epoch随机打乱（训练集），防止模型记住顺序
      - pin_memory=True: 锁页内存（加速CPU→GPU传输）
      - drop_last=True: 丢弃最后不完整的batch（保持batch_size一致）
      - num_workers=0: 单进程加载（Windows下多进程可能有问题）
    """
    n = len(src_texts)
    n_val = max(100, int(n * val_ratio))
    n_train = n - n_val

    # ── 划分 ──
    train_src = src_texts[:n_train]
    train_tgt = tgt_texts[:n_train]
    val_src = src_texts[n_train:n_train + n_val]
    val_tgt = tgt_texts[n_train:n_train + n_val]

    print(f"[Data] 训练集: {n_train} 句对, 验证集: {n_val} 句对")

    # ── 创建 Dataset ──
    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer, max_len)

    # ── 创建 DataLoader ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx=0, max_len=max_len),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,           # 验证集不需要打乱
        collate_fn=lambda b: collate_fn(b, pad_idx=0, max_len=max_len),
        num_workers=0,
        pin_memory=True,
        drop_last=False,         # 验证集保留所有数据
    )

    return train_loader, val_loader


# ==============================================================================
# 6. 主流程入口
# ==============================================================================

def prepare_data():
    """
    【数据准备主流程 — main.py 调用的唯一入口】

    【流程】
      load_opus100() → filter_data() → build_vocab() → create_dataloaders()
      ↑ 加载5万条      ↑ 过滤剩余4.6万  ↑ 构建1万词表   ↑ 创建DataLoader

    【返回】
      train_loader: 训练DataLoader (710 batches × 64)
      val_loader:   验证DataLoader (15 batches × 64)
      tokenizer:    训练好的SimpleTokenizer
    """
    print("=" * 60)
    print("  数据预处理 Pipeline")
    print("=" * 60)

    # 1. 加载
    src_texts, tgt_texts = load_opus100(
        config.data_dir,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        max_samples=config.max_train_samples,
    )

    # 2. 清洗过滤
    src_texts, tgt_texts = filter_data(
        src_texts, tgt_texts,
        min_len=config.min_seq_len,
        max_len=config.max_seq_len,
    )

    # 3. 构建词表
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(src_texts, tgt_texts)
    tokenizer.save(config.tokenizer_path)

    # 4. 创建 DataLoader
    train_loader, val_loader = create_dataloaders(
        src_texts, tgt_texts, tokenizer,
        batch_size=config.batch_size,
        max_len=config.max_seq_len,
        val_ratio=0.02,
    )

    print(f"[Data] DataLoader 创建完毕")
    print(f"[Data]   训练批次数: {len(train_loader)}")
    print(f"[Data]   验证批次数: {len(val_loader)}")
    print(f"[Data]   词表大小:   {len(tokenizer)}")
    print("=" * 60)

    return train_loader, val_loader, tokenizer


# ── 直接运行此文件可独立测试数据模块 ──
if __name__ == "__main__":
    train_loader, val_loader, tokenizer = prepare_data()
    # 取一个batch看看
    src, src_mask, tgt_in, tgt_mask, tgt_out = next(iter(train_loader))
    print(f"\n[测试 Batch]")
    print(f"  src 形状:      {src.shape}")
    print(f"  src_mask 形状: {src_mask.shape}")
    print(f"  tgt_input:     {tgt_in.shape}")
    print(f"  tgt_output:    {tgt_out.shape}")
    print(f"  src[0] IDs:    {src[0].tolist()}")
    print(f"  tgt_in[0] IDs: {tgt_in[0].tolist()}")
