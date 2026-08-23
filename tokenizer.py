
import json      
import re        
import os        
from collections import Counter   
from typing import List, Tuple, Optional  

import torch   


class SimpleTokenizer:
    
    # 特殊Token定义 

    PAD_TOKEN = "<pad>"   # 填充，让批次内不等长句子对齐
    SOS_TOKEN = "<sos>"   # 句子开头，模型知道开始
    EOS_TOKEN = "<eos>"   # 句子结束，模型可以停
    UNK_TOKEN = "<unk>"   # 词表里没有的词


    PAD_IDX = 0   # torch.full(fill_value=0) 默认填充0
    SOS_IDX = 1   # 每个编码句子都以 SOS 开头
    EOS_IDX = 2   # 每个编码句子都以 EOS 结尾
    UNK_IDX = 3   # 遇到词表外词汇时使用



    def __init__(self, vocab_size: int = 10000):

        self.vocab_size = vocab_size

        # 两个字典互为反函数 
        self.stoi = {
            self.PAD_TOKEN: self.PAD_IDX,  # <pad> → 0
            self.SOS_TOKEN: self.SOS_IDX,  # <sos> → 1
            self.EOS_TOKEN: self.EOS_IDX,  # <eos> → 2
            self.UNK_TOKEN: self.UNK_IDX,  # <unk> → 3
        }

        self.itos = {
            self.PAD_IDX: self.PAD_TOKEN,  # 0 → <pad>
            self.SOS_IDX: self.SOS_TOKEN,  # 1 → <sos>
            self.EOS_IDX: self.EOS_TOKEN,  # 2 → <eos>
            self.UNK_IDX: self.UNK_TOKEN,  # 3 → <unk>
        }
        self.zh_offset: Optional[int] = None
        # 中文词表占用的字符数    
        self.en_start_id: Optional[int] = None# 英文token的起始ID = 4（跳过4个特殊token） + 中文


    @staticmethod
    def tokenize_en(text: str) -> List[str]:


        text = text.strip().lower()


        text = re.sub(

            r"([.,!?;:\"'()\[\]{}<>/\\\-@#$%^&*+=~`|])",
            r" \1 ",
            text
        )

        tokens = text.split()

        return tokens


    @staticmethod
    def tokenize_zh(text: str) -> List[str]:


        text = text.strip()

        tokens = [c for c in text if not c.isspace()]

        return tokens



    def build_vocab(self, zh_texts: List[str], en_texts: List[str]):



        # 统计中文汉字频率 
        zh_counter = Counter()
        for text in zh_texts:
            zh_counter.update(self.tokenize_zh(text))

        #统计英文单词频率 
        en_counter = Counter()
        for text in en_texts:

            en_counter.update(self.tokenize_en(text))


        available = self.vocab_size - 4         # 10000-4 = 9996

        zh_budget = available // 2               #  9996//2 = 4998
        en_budget = available - zh_budget        #  9996-4998 = 4998

        # 取高频中文字符 

        zh_top = zh_counter.most_common(zh_budget)

        for i, (char, _) in enumerate(zh_top):
            idx = 4 + i                          # 中文起始ID = 4
            self.stoi[char] = idx                
            self.itos[idx] = char                

        # 计算英文token起始ID 

        self.zh_offset = len(zh_top)              # 例如 4998
        self.en_start_id = 4 + self.zh_offset     # 例如 4+4998 = 5002

        # 取高频英文单词 
        en_top = en_counter.most_common(en_budget)
        for i, (word, _) in enumerate(en_top):
            idx = self.en_start_id + i            # 例如 5002+i
            self.stoi[word] = idx                 
            self.itos[idx] = word                 


        print(f"  中文词表一共{len(zh_top)} 个词")
        print(f"  英文词表一共 {len(en_top)} 个词")



    def encode_zh(self, text: str, add_special: bool = True) -> List[int]:

        # 按字符拆开
        tokens = self.tokenize_zh(text)

        #  查字典：stoi.get(token, default)

        ids = [self.stoi.get(t, self.UNK_IDX) for t in tokens]

        # 3. 加首尾标记
        if add_special:
            # 在开头插入 SOS，在结尾追加 EOS
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]

        return ids

    def encode_en(self, text: str, add_special: bool = True) -> List[int]:

        #按单词+标点拆开
        tokens = self.tokenize_en(text)

        #查字典
        ids = [self.stoi.get(t, self.UNK_IDX) for t in tokens]

        # 加首尾标记
        if add_special:
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]

        return ids


    def decode(self, ids: List[int], skip_special: bool = True) -> str:

        # 跳过特殊token
        specials = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX, self.UNK_IDX}

        tokens = []
        for idx in ids:
            if skip_special and idx in specials:
                continue

            token = self.itos.get(idx, self.UNK_TOKEN)

            if token != self.UNK_TOKEN:
                tokens.append(token)

        text = " ".join(tokens)

        text = re.sub(r" ([.,!?;:)'\"])", r"\1", text)

        text = re.sub(r"([(]) ", r"\1", text)

        return text


    def encode_batch_zh(self, texts: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_ids = []
        for text in texts:
            # 逐条编码（含SOS/EOS）
            ids = self.encode_zh(text, add_special=True)
            # 截断到 max_len
            ids = ids[:max_len]
            batch_ids.append(ids)


        return self._pad_batch(batch_ids, max_len)

    def encode_batch_en(self, texts: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_ids = []
        for text in texts:
            ids = self.encode_en(text, add_special=True)
            ids = ids[:max_len]
            batch_ids.append(ids)

        return self._pad_batch(batch_ids, max_len)

    @staticmethod
    def _pad_batch(batch_ids: List[List[int]], max_len: int):

        batch_size = len(batch_ids)


        padded = torch.full(
            (batch_size, max_len),
            fill_value=0,         
            dtype=torch.long       
        )


        mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool      
        )


        for i, ids in enumerate(batch_ids):
            # 取实际长度不能超过max_len
            length = min(len(ids), max_len)

            # 把这条句子的token ID复制到 padded 的第i行前length列
            # 后面的列保持0（PAD）
            padded[i, :length] = torch.tensor(
                ids[:length],
                dtype=torch.long
            )

            # 把 mask 的前length列设为True，
            mask[i, :length] = True

        return padded, mask



    def save(self, path: str):
  
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "zh_offset": self.zh_offset,
            "en_start_id": self.en_start_id,
        }


        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f" 词表保存到 {path}")

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
    
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)


        tokenizer = cls(vocab_size=data["vocab_size"])

        tokenizer.stoi = data["stoi"]


        if "itos" in data:

            tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        else:

            tokenizer.itos = {
                int(v) if isinstance(v, str) else v: k
                for k, v in data["stoi"].items()
            }


        tokenizer.zh_offset = data.get("zh_offset")
        tokenizer.en_start_id = data.get("en_start_id")


        if not tokenizer.itos or len(tokenizer.itos) < len(tokenizer.stoi):
            tokenizer.itos = {
                int(v) if isinstance(v, str) else v: k
                for k, v in tokenizer.stoi.items()
            }

        print(f"词表加载了: {len(tokenizer.stoi)} 个词儿")
        return tokenizer

    def __len__(self):

        return len(self.stoi)
