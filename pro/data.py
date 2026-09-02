
import os
import time
import re
from typing import Tuple, Optional, List
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pyarrow.ipc as ipc   # Apache Arrow IPC: 读取.arrow文件

from config import config
from tokenizer import BpeTokenizer


# Windows 下 num_workers 的注意点
# num_workers>0 会开子进程并行编码。Windows 用的是 spawn 方式，每个 worker 会
# 重新 import 本模块并 pickle 一次 dataset，所以：
#   1) 数据加载必须写在函数里（不能写在模块顶层），否则会被重复执行；
#   2) dataset 里只存"文本 + tokenizer 路径"，tokenizer 对象在 worker 内惰性加载，
#      避免 sentencepiece 的 C++ 对象 pickle 失败。
# 注意：Windows 下 num_workers>0 时，每个 worker 都是一个独立的 Python 进程，
# 会各自重新 import torch（每个约 1.5GB）+ pickle 一份文本。本机 16GB 内存，
# 开 2 个 worker（约 3GB 额外）安全，能并行编码、隐藏数据加载时间。之前开 4
# 会爆内存（4 进程 + pickle 峰值 + 同时开着的其他程序），所以定在 2，别往上加。
NUM_WORKERS = 2


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_opus100_stream(data_dir: str, src_lang: str = "zh", tgt_lang: str = "en",
                        max_samples: Optional[int] = None,
                        min_len: int = 2, max_len: int = 128
                        ) -> Tuple[List[str], List[str]]:
    """流式加载 + 边读边过滤。

    原来的 load_opus100 + filter_data 是"先全量读进内存，再过滤"，100 万条时
    会同时占两份内存（原始 + 清洗后），8G 内存直接崩。这里改成读一条滤一条，
    只保留最终清洗后的文本，峰值内存减半。
    """
    arrow_dir = os.path.join(data_dir, "en-zh", "0.0.0")
    subdirs = [d for d in os.listdir(arrow_dir)
               if os.path.isdir(os.path.join(arrow_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"未找到 Arrow 文件目录: {arrow_dir}")
    arrow_dir = os.path.join(arrow_dir, subdirs[0])
    train_file = os.path.join(arrow_dir, "opus-100-train.arrow")

    src_texts: List[str] = []
    tgt_texts: List[str] = []

    with ipc.open_stream(train_file) as reader:
        for batch in reader:
            translations = batch.column("translation").to_pylist()
            for row in translations:
                if src_lang == "zh":
                    src, tgt = row["zh"], row["en"]
                else:
                    src, tgt = row["en"], row["zh"]

                # ── 边读边过滤（和原 filter_data 一致，但去掉了去重以省内存）──
                src = clean_text(src)
                tgt = clean_text(tgt)
                if not src or not tgt:                 # 空句子
                    continue
                if len(src) < min_len or len(tgt) < min_len:
                    continue
                if len(src) > max_len * 5 or len(tgt) > max_len * 5:
                    continue
                ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
                if ratio > 10:                          # 长度比太悬殊
                    continue

                src_texts.append(src)
                tgt_texts.append(tgt)

                if max_samples and len(src_texts) >= max_samples:
                    break
            if max_samples and len(src_texts) >= max_samples:
                break

    print(f"加载并过滤完成，共 {len(src_texts)} 个训练句对")
    return src_texts, tgt_texts


class TranslationDataset(Dataset):
    # 惰性编码：__init__ 只存文本，__getitem__ 里才编码。

    # 原来的写法是在 __init__ 里把所有句对都 encode 成 id 列表存进内存。
    # 100 万条 × 平均 30 token 的 Python list 对象会爆内存。改成惰性后，
    # __init__ 不持有任何 token id，编码在取用时发生，由 DataLoader 的
    # prefetch 控制节奏。
    

    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 tokenizer_path: str, max_len: int):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer_path = tokenizer_path
        self.max_len = max_len
        # 每个 worker 进程各自惰性加载一次 tokenizer（见 _get_tokenizer）
        self._tokenizer = None

    def _get_tokenizer(self) -> BpeTokenizer:
        if self._tokenizer is None:
            self._tokenizer = BpeTokenizer.from_file(self.tokenizer_path)
        return self._tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        tokenizer = self._get_tokenizer()

        src_ids = tokenizer.encode_zh(self.src_texts[idx], add_special=True)
        src_ids = src_ids[:self.max_len]

        tgt_ids = tokenizer.encode_en(self.tgt_texts[idx], add_special=True)
        tgt_ids = tgt_ids[:self.max_len]

        # 极短句子可能 encode 完只剩 <s> 或 <s></s>，兜底补成最小合法序列
        if len(tgt_ids) < 2:
            tgt_ids = [tokenizer.SOS_IDX, tokenizer.EOS_IDX]

        tgt_input = tgt_ids[:-1]    # 去掉最后的 EOS
        tgt_output = tgt_ids[1:]    # 去掉最前的 SOS

        return src_ids, tgt_input, tgt_output


def collate_fn(batch, pad_idx: int = 0, max_len: int = 128):
    B = len(batch)

    # 找 batch 内最大长度（动态 padding，不整批 pad 到 max_len）
    max_src = min(max(len(d[0]) for d in batch), max_len)
    max_tgt = min(max(len(d[1]) for d in batch), max_len)

    src = torch.full((B, max_src), fill_value=pad_idx, dtype=torch.long)
    src_mask = torch.zeros((B, max_src), dtype=torch.bool)

    tgt_input = torch.full((B, max_tgt), fill_value=pad_idx, dtype=torch.long)
    tgt_mask = torch.zeros((B, max_tgt), dtype=torch.bool)
    tgt_output = torch.full((B, max_tgt), fill_value=pad_idx, dtype=torch.long)

    for i, (s, ti, to) in enumerate(batch):
        s_len = min(len(s), max_src)
        src[i, :s_len] = torch.tensor(s[:s_len], dtype=torch.long)
        src_mask[i, :s_len] = True

        t_len = min(len(ti), max_tgt)
        tgt_input[i, :t_len] = torch.tensor(ti[:t_len], dtype=torch.long)
        tgt_mask[i, :t_len] = True

        o_len = min(len(to), max_tgt)
        tgt_output[i, :o_len] = torch.tensor(to[:o_len], dtype=torch.long)

    return src, src_mask, tgt_input, tgt_mask, tgt_output


class LengthBucketSampler(Sampler):
    # 动态 batching：按源句长度分桶，同一 batch 内长度接近，减少 padding 浪费。

    # 原理：每个 epoch 先把所有样本按长度排序 → 切成 batch_size 大小的块 →
    # 块与块之间 shuffle。这样既保留了随机性（不同 epoch 顺序不同），又让每个
    # batch 内的句子长度相近，动态 padding 时浪费最少。对 100 万条长尾数据，
    # 这能显著减少算在 <pad> 上的无效计算。
    

    def __init__(self, src_lengths: List[int], batch_size: int):
        self.src_lengths = src_lengths
        self.batch_size = batch_size

    def __iter__(self):
        import random
        indices = list(range(len(self.src_lengths)))
        indices.sort(key=lambda i: self.src_lengths[i])          # 按长度排序
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        random.shuffle(batches)                                   # 块间 shuffle
        for b in batches:
            yield from b

    def __len__(self):
        return len(self.src_lengths)


def create_dataloaders(src_texts: List[str], tgt_texts: List[str],
                       tokenizer_path: str,
                       batch_size: int = 64, max_len: int = 128,
                       val_ratio: float = 0.02, num_workers: int = NUM_WORKERS):

    n = len(src_texts)
    n_val = max(100, int(n * val_ratio))
    n_train = n - n_val

    train_src = src_texts[:n_train]
    train_tgt = tgt_texts[:n_train]
    val_src = src_texts[n_train:n_train + n_val]
    val_tgt = tgt_texts[n_train:n_train + n_val]

    print(f"训练集 {n_train} 句对, 验证集 {n_val} 句对")

    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer_path, max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer_path, max_len)

    # 训练集：长度分桶采样（动态 batching）
    train_sampler = LengthBucketSampler(
        [len(s) for s in train_src], batch_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,       # 用 sampler 控制顺序，shuffle 必须为 False
        collate_fn=partial(collate_fn, pad_idx=0, max_len=max_len),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),   # 复用 worker，避免每个 epoch 重新 spawn
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,               # 验证集不需要打乱
        collate_fn=partial(collate_fn, pad_idx=0, max_len=max_len),
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def prepare_data():
    src_texts, tgt_texts = load_opus100_stream(
        config.data_dir,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        max_samples=config.max_train_samples,
        min_len=config.min_seq_len,
        max_len=config.max_seq_len,
    )

    # BPE 分词器是提前训练好、存成 spm.model 的（跑 train_tokenizer.py 生成）。
    # 这里只加载，不再像 SimpleTokenizer 那样现场 build_vocab。
    if not os.path.exists(config.tokenizer_path):
        raise FileNotFoundError(
            f"没找到 BPE 模型 {config.tokenizer_path}\n"
            f"请先运行:  python train_tokenizer.py"
        )

    # 加载 BPE 分词器，返回给 train.py 用于 BLEU 评估（decode 参考/候选译文）
    tokenizer = BpeTokenizer.from_file(config.tokenizer_path)

    train_loader, val_loader = create_dataloaders(
        src_texts, tgt_texts,
        tokenizer_path=config.tokenizer_path,
        batch_size=config.batch_size,
        max_len=config.max_seq_len,
        val_ratio=0.02,
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print("=" * 60)

    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    train_loader, val_loader, _ = prepare_data()
    src, src_mask, tgt_in, tgt_mask, tgt_out = next(iter(train_loader))
    print(f"  src 形状:      {src.shape}")
    print(f"  src_mask 形状: {src_mask.shape}")
    print(f"  tgt_input:     {tgt_in.shape}")
    print(f"  tgt_output:    {tgt_out.shape}")
