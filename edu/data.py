
import os
import time
import re
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.ipc as ipc   # Apache Arrow IPC: 读取.arrow文件

from config import config
from tokenizer import SimpleTokenizer




def load_opus100(data_dir: str, src_lang: str = "zh", tgt_lang: str = "en",
                 max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:

    # Arrow文件路径
    arrow_dir = os.path.join(data_dir, "en-zh", "0.0.0")

    # HuggingFace datasets 用 hash 命名子目录
    subdirs = [d for d in os.listdir(arrow_dir)
               if os.path.isdir(os.path.join(arrow_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"未找到 Arrow 文件目录: {arrow_dir}")

    arrow_dir = os.path.join(arrow_dir, subdirs[0])
    train_file = os.path.join(arrow_dir, "opus-100-train.arrow")


    src_texts = []
    tgt_texts = []


    with ipc.open_stream(train_file) as reader:

        for batch_idx, batch in enumerate(reader):

            translations = batch.column("translation").to_pylist()

            for row in translations:

                if src_lang == "zh":
                    src_texts.append(row["zh"])
                    tgt_texts.append(row["en"])
                else:
                    src_texts.append(row["en"])
                    tgt_texts.append(row["zh"])

                # 达到上限就停止
                if max_samples and len(src_texts) >= max_samples:
                    break

            if max_samples and len(src_texts) >= max_samples:
                break

    print(f"加载了 {len(src_texts)} 个训练句对")
    return src_texts, tgt_texts




def clean_text(text: str) -> str:

    return re.sub(r"\s+", " ", text).strip()


def filter_data(src_texts: List[str], tgt_texts: List[str],
                min_len: int = 2, max_len: int = 64) -> Tuple[List[str], List[str]]:

    print(f" 过滤前 {len(src_texts)} 个句对")

    seen = set()               # 去重用
    clean_src, clean_tgt = [], []

    for src, tgt in zip(src_texts, tgt_texts):
        # 清洗空白
        src = clean_text(src)
        tgt = clean_text(tgt)

        # 过滤空句子 
        if not src or not tgt:
            continue

        #  过滤长度比异常 
        if len(src) < min_len or len(tgt) < min_len:
            continue
        if len(src) > max_len * 5 or len(tgt) > max_len * 5:
            continue
        ratio = max(len(src), len(tgt)) / max(1, min(len(src), len(tgt)))
        if ratio > 10:
            continue

        #  去重 
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)

        clean_src.append(src)
        clean_tgt.append(tgt)

    removed = len(src_texts) - len(clean_src)
    print(f" 过滤后还剩 {len(clean_src)} 个句对 ")
    return clean_src, clean_tgt


class TranslationDataset(Dataset):


    def __init__(self, src_texts: List[str], tgt_texts: List[str],
                 tokenizer: SimpleTokenizer, max_len: int):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []

        print(f" 编码 {len(src_texts)} 条数据 ")
        start = time.time()

        for src, tgt in zip(src_texts, tgt_texts):
            #  编码源语言
            src_ids = tokenizer.encode_zh(src, add_special=True)
            src_ids = src_ids[:max_len]   # 截断

            # 编码目标语言 
            tgt_ids = tokenizer.encode_en(tgt, add_special=True)
            tgt_ids = tgt_ids[:max_len]


            if len(src_ids) < 2 or len(tgt_ids) < 2:
                continue

            tgt_input = tgt_ids[:-1]    # 去掉最后的EOS
            tgt_output = tgt_ids[1:]    # 去掉最前的SOS

            self.data.append({
                "src": src_ids,
                "tgt_input": tgt_input,
                "tgt_output": tgt_output,
            })

        print(f" 编码完成 {len(self.data)} 条, ")


    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return self.data[idx]



def collate_fn(batch: List[dict], pad_idx: int = 0, max_len: int = 64):

    B = len(batch)

    # 找batch内最大长度
    max_src = min(max(len(d["src"]) for d in batch), max_len)
    max_tgt = min(max(len(d["tgt_input"]) for d in batch), max_len)

    #  创建全零矩阵 
    src = torch.full((B, max_src), fill_value=pad_idx, dtype=torch.long)
    src_mask = torch.zeros((B, max_src), dtype=torch.bool)

    tgt_input = torch.full((B, max_tgt), fill_value=pad_idx, dtype=torch.long)
    tgt_mask = torch.zeros((B, max_tgt), dtype=torch.bool)
    tgt_output = torch.full((B, max_tgt), fill_value=pad_idx, dtype=torch.long)

    #  逐条填入
    for i, d in enumerate(batch):

        s_len = min(len(d["src"]), max_src)
        src[i, :s_len] = torch.tensor(d["src"][:s_len], dtype=torch.long)
        src_mask[i, :s_len] = True


        t_len = min(len(d["tgt_input"]), max_tgt)
        tgt_input[i, :t_len] = torch.tensor(d["tgt_input"][:t_len], dtype=torch.long)
        tgt_mask[i, :t_len] = True


        o_len = min(len(d["tgt_output"]), max_tgt)
        tgt_output[i, :o_len] = torch.tensor(d["tgt_output"][:o_len], dtype=torch.long)

    return src, src_mask, tgt_input, tgt_mask, tgt_output



def create_dataloaders(src_texts: List[str], tgt_texts: List[str],
                       tokenizer: SimpleTokenizer,
                       batch_size: int = 64, max_len: int = 64,
                       val_ratio: float = 0.02):

    n = len(src_texts)
    n_val = max(100, int(n * val_ratio))
    n_train = n - n_val


    train_src = src_texts[:n_train]
    train_tgt = tgt_texts[:n_train]
    val_src = src_texts[n_train:n_train + n_val]
    val_tgt = tgt_texts[n_train:n_train + n_val]

    print(f"训练集 {n_train} g1句对, 验证集{n_val}g1 句对")


    train_dataset = TranslationDataset(train_src, train_tgt, tokenizer, max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, tokenizer, max_len)


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



def prepare_data():


    src_texts, tgt_texts = load_opus100(
        config.data_dir,
        src_lang=config.src_lang,
        tgt_lang=config.tgt_lang,
        max_samples=config.max_train_samples,
    )


    src_texts, tgt_texts = filter_data(
        src_texts, tgt_texts,
        min_len=config.min_seq_len,
        max_len=config.max_seq_len,
    )


    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(src_texts, tgt_texts)
    tokenizer.save(config.tokenizer_path)

    train_loader, val_loader = create_dataloaders(
        src_texts, tgt_texts, tokenizer,
        batch_size=config.batch_size,
        max_len=config.max_seq_len,
        val_ratio=0.02,
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    print("=" * 60)

    return train_loader, val_loader, tokenizer



if __name__ == "__main__":
    train_loader, val_loader, tokenizer = prepare_data()
    # 取一个batch看看
    src, src_mask, tgt_in, tgt_mask, tgt_out = next(iter(train_loader))
    # print(f"  src 形状:      {src.shape}")
    # print(f"  src_mask 形状: {src_mask.shape}")
    # print(f"  tgt_input:     {tgt_in.shape}")
    # print(f"  tgt_output:    {tgt_out.shape}")
    # print(f"  src[0] IDs:    {src[0].tolist()}")
    # print(f"  tgt_in[0] IDs: {tgt_in[0].tolist()}")
