"""
================================================================================
Transformer 教育版 — 分词器模块 (Tokenizer)
================================================================================

【本模块在Pipeline中的位置】
  原始文本 → [tokenizer.py] → token ID 序列 → Embedding层

【为什么需要分词器】
  神经网络只能处理数字。分词器的任务就是把人类可读的文本（中文字符串、
  英文单词串）转换成模型能接收的整数ID序列，以及把ID序列还原成文本。

【分词策略】
  - 中文：字符级（character-level），每个汉字 = 1个token
    原因：中文天然以字为基本单位，字与字之间无空格分隔，字符级最简单直观
    例："你好" → ["你","好"] → [ID(你), ID(好)]

  - 英文：词级（word-level），按空格+标点切分
    原因：英文天然以空格分词，词级足够表达语义
    例："Hello, world!" → ["hello",",","world","!"] → [ID(hello), ...]

  - 特殊Token（固定ID，所有词表共享）:
    PAD = 0  → 填充符，batch中短句补长，loss计算时跳过
    SOS = 1  → 句子起始 (Start Of Sequence)，每个句子开头加
    EOS = 2  → 句子结束 (End Of Sequence)，每个句子结尾加，推理时遇到就停止
    UNK = 3  → 未知词 (Unknown)，词表中不存在的词统一映射到此

【联合词表（Shared Vocabulary）】
  中文和英文共享同一个ID空间，模型的Embedding矩阵同时服务两种语言。
  这样做的好处：
    1. 只需一个 Embedding 矩阵（参数更少）
    2. 编码器和解码器共享同一个词表
    3. 训练和推理更简单

  词表布局：
    ID 0-3:   特殊Token (PAD, SOS, EOS, UNK)
    ID 4-N:   中文高频字符 (~5000个)
    ID N+1-M: 英文高频单词 (~5000个)
  其中 N = 4 + zh_vocab_size, M = vocab_size

【使用示例】
  tokenizer = SimpleTokenizer(vocab_size=10000)
  tokenizer.build_vocab(zh_sentences, en_sentences)  # 统计词频构建词表

  # 编码（文本 → ID序列）
  ids_zh = tokenizer.encode_zh("你好")        # → [1, ID(你), ID(好), 2]
  ids_en = tokenizer.encode_en("hello")      # → [1, ID(hello), 2]

  # 解码（ID序列 → 文本）
  text = tokenizer.decode(ids_en)            # → "hello"

  # 批处理（自动padding到相同长度）
  src, mask = tokenizer.encode_batch_zh(["你好", "世界"], max_len=16)
  # src = [[1,ID(你),ID(好),2, 0,0,...], [1,ID(世),ID(界),2, 0,0,...]]
  # mask= [[T, T,     T,    T, F,F,...], [T, T,     T,    T, F,F,...]]
================================================================================
"""

import json       # 读写词表文件（JSON格式，人类可读）
import re         # 正则表达式，英文分词时处理标点符号
import os         # 创建保存目录
from collections import Counter   # 统计词频
from typing import List, Tuple, Optional   # 类型注解

import torch      # 张量操作（创建padded tensor）


# ==============================================================================
# SimpleTokenizer 类
# ==============================================================================
class SimpleTokenizer:
    """
    简易联合分词器 — 中文用字符级，英文用词级

    【设计思想】
      分词器的核心就是两个字典（Python dict）：
        stoi (string-to-index):  "你" → 7, "hello" → 18
        itos (index-to-string):  7 → "你", 18 → "hello"
      再加上编码/解码逻辑。简单但不简陋。

    【数据流】
      编码: text → tokenize → list[str] → stoi lookup → list[int]
      解码: list[int] → itos lookup → list[str] → join → text
    """

    # ── 特殊Token定义 ─────────────────────────────────────────────
    # 字符串形式（用于保存词表时的人类可读标签）
    PAD_TOKEN = "<pad>"   # 填充，让批次内不等长句子对齐
    SOS_TOKEN = "<sos>"   # 句子开头，模型知道"开始生成"
    EOS_TOKEN = "<eos>"   # 句子结束，模型知道"可以停了"
    UNK_TOKEN = "<unk>"   # 未登录词，词表里没有的词都用这个代替

    # 整数ID（模型实际接收的数字）
    PAD_IDX = 0   # 为什么PAD=0？因为 torch.full(fill_value=0) 默认填充0
    SOS_IDX = 1   # 每个编码句子都以 SOS 开头
    EOS_IDX = 2   # 每个编码句子都以 EOS 结尾
    UNK_IDX = 3   # 遇到词表外词汇时使用（中文生僻字、英文罕见词）

    # ── 构造函数 ───────────────────────────────────────────────────

    def __init__(self, vocab_size: int = 10000):
        """
        初始化分词器，只创建特殊token，不构建词表。

       【参数说明】
          vocab_size: 联合词表总大小，默认10000
                      包含了4个特殊token + 中文token + 英文token
                      为什么默认10000？
                        - 教育版用50000条数据，10000词表足够覆盖高频字词
                        - 太小→UNK太多，太大→Embedding矩阵浪费显存

        【初始化时的状态】
          初始化后 stoi 和 itos 只包含4个特殊token。
          需要调用 build_vocab() 才能用于实际的编码解码。
        """
        self.vocab_size = vocab_size

        # ── 核心数据结构：两个字典互为反函数 ──
        # stoi: string to index，"你好"这个词在中文tokenize后每个字独立查找
        #       比如 stoi["你"] = 7, stoi["hello"] = 18
        self.stoi = {
            self.PAD_TOKEN: self.PAD_IDX,  # "<pad>" → 0
            self.SOS_TOKEN: self.SOS_IDX,  # "<sos>" → 1
            self.EOS_TOKEN: self.EOS_IDX,  # "<eos>" → 2
            self.UNK_TOKEN: self.UNK_IDX,  # "<unk>" → 3
        }

        # itos: index to string，解码时用
        #       比如 itos[7] = "你", itos[18] = "hello"
        self.itos = {
            self.PAD_IDX: self.PAD_TOKEN,  # 0 → "<pad>"
            self.SOS_IDX: self.SOS_TOKEN,  # 1 → "<sos>"
            self.EOS_IDX: self.EOS_TOKEN,  # 2 → "<eos>"
            self.UNK_IDX: self.UNK_TOKEN,  # 3 → "<unk>"
        }

        # 中文词表占用的字符数（用于计算英文token的起始ID）
        # 在 build_vocab() 中设置
        self.zh_offset: Optional[int] = None

        # 英文token的起始ID = 4（跳过4个特殊token） + zh_offset
        # 例如：zh_offset=4998 → en_start_id=5002
        self.en_start_id: Optional[int] = None

    # ==========================================================================
    # 英文分词：按空格和标点切分
    # ==========================================================================

    @staticmethod
    def tokenize_en(text: str) -> List[str]:
        """
        英文分词 —— 把英文句子拆成单词和标点列表。

        【分词策略】
          - 先全部转小写（Hello → hello），避免大小写分散词频
          - 在标点前后插入空格，然后按空格split
          - 标点作为独立token（"hello," → ["hello", ","]）
            为什么？标点独立可以让模型学到标点和词的独立语义

        【正则表达式详解】
          r"([.,!?;:\"'()\[\]{}<>/\\\-@#$%^&*+=~`|])"
          这个正则会匹配任何一个英文标点符号，括号 () 表示"捕获组"
          r" \1 " 表示：把匹配到的标点替换为 "空格+标点+空格"
          然后 .split() 按空白字符切分，标点自然成为独立token

        【输入输出示例】
          "Hello, world!"              → ["hello", ",", "world", "!"]
          "Don't do that."             → ["don", "'", "t", "do", "that", "."]
          "It's a state-of-the-art AI."→ ["it", "'", "s", "a", "state", "-", "of",
                                           "-", "the", "-", "art", "ai", "."]
          （注意：连字符也被当作标点独立出来了）

        【局限】
          - "n't" 没被特殊处理 → "don't" 变成 don/'/t，比较碎片化
          - 生产版用 BPE（SentencePiece）可以自动学习更好的子词切分
        """
        # 1. strip(): 去除首尾空白
        # 2. lower(): 统一转小写，避免 Hello/hello 被当作两个不同词
        text = text.strip().lower()

        # 3. re.sub: 在每一个标点符号前后各加一个空格
        #    正则 ( ) 是捕获组，\1 引用捕获到的标点符号
        #    例如 "hello," → "hello , "，然后 split 就变成了 ["hello", ","]
        #    注意：这个正则涵盖了英文中几乎所有的标点符号
        text = re.sub(
            # 捕获组：匹配任意一个英文标点
            r"([.,!?;:\"'()\[\]{}<>/\\\-@#$%^&*+=~`|])",
            # 替换为 "空格 + 该标点 + 空格"
            r" \1 ",
            text
        )

        # 4. split(): 按任意空白字符（空格、tab、换行等）切分
        #    连续的多个空格会被忽略（split() 默认行为）
        tokens = text.split()

        return tokens

    # ==========================================================================
    # 中文分词：按字符切分
    # ==========================================================================

    @staticmethod
    def tokenize_zh(text: str) -> List[str]:
        """
        中文分词 —— 每个汉字 = 1个token。

        【为什么中文用字符级】
          中文不像英文有天然的空格分隔，词边界模糊。
          "中华人民共和国" → 是一个词还是"中华/人民/共和国"？
          如果先做中文分词，依赖额外的分词工具（如jieba），且错误会传播。
          字符级分词虽然token数更多，但：
            1. 100%确定，没有分词歧义
            2. 不需要额外依赖
            3. 对于教育版够用了
          生产版会用 SentencePiece BPE 自动学习子词切分。

        【过滤策略】
          只过滤空白字符（空格、全角空格、换行等），保留标点符号。
          中文标点（，。！？）也是token，帮助模型理解句子结构。

        【输入输出示例】
          "你好世界"    → ["你", "好", "世", "界"]
          "你好，世界！" → ["你", "好", "，", "世", "界", "！"]
          "Hello世界"   → ["H", "e", "l", "l", "o", "世", "界"]
          （注意：拉丁字母也会被当作独立的token，但这在联合词表里会被映射到
           英文词表那边，所以问题不大。实际语料中中英混用很少见）
        """
        # 1. strip(): 去除首尾空白
        text = text.strip()

        # 2. 列表推导式：遍历每个字符，过滤掉空白字符
        #    c.isspace() 判断是否空白（空格、\t、\n、\r、全角空格等）
        #    注意：不调用 lower()！中文没有大小写概念
        tokens = [c for c in text if not c.isspace()]

        return tokens

    # ==========================================================================
    # 构建词表：从语料中统计频率，选出最高频的字/词
    # ==========================================================================

    def build_vocab(self, zh_texts: List[str], en_texts: List[str]):
        """
        【词表构建的核心流程】
          1. 遍历所有中文句子，对每个汉字计数
          2. 遍历所有英文句子，对每个单词计数
          3. 预算分配：一半ID给中文，一半给英文
          4. 按频率从高到低取top-K，分配ID
          5. 生成 stoi 和 itos 字典

        【为什么要按频率取top-K】
          自然语言的词频服从齐夫定律（Zipf's law）：
            排名第1的词出现次数 ≈ 排名第2的2倍 ≈ 排名第3的3倍...
          绝大多数词只出现一两次，对训练没有帮助。
          取 top-5000 高频字/词 → 覆盖 >95% 的语料 → UNK 率 <5%

        【词表布局示例（vocab_size=10000）】
          ID 0:     <pad>    填充
          ID 1:     <sos>    起始
          ID 2:     <eos>    结束
          ID 3:     <unk>    未知
          ID 4-5001: 中文高频字符（4998个）
          ID 5002-9999: 英文高频单词（4998个）

        【参数说明】
          zh_texts: 所有中文句子（训练集 + 验证集）
          en_texts: 所有英文句子
          一般在 data.py 的 prepare_data() 中调用，此时数据已清洗完毕
        """
        print(f"[Tokenizer] 构建词表 (vocab_size={self.vocab_size})...")

        # ── 步骤1：统计中文汉字频率 ──
        # Counter 是一个特殊的字典，key=字符, value=出现次数
        # .update() 会把列表中的每个元素计数+1
        zh_counter = Counter()
        for text in zh_texts:
            # tokenize_zh 把句子拆成字符列表 → Counter 自动统计
            zh_counter.update(self.tokenize_zh(text))

        # ── 步骤2：统计英文单词频率 ──
        en_counter = Counter()
        for text in en_texts:
            # tokenize_en 把句子拆成单词列表 → Counter 自动统计
            en_counter.update(self.tokenize_en(text))

        # ── 步骤3：计算分配预算 ──
        # 总共 vocab_size 个位置，扣除4个特殊token
        available = self.vocab_size - 4         # 例如 10000-4 = 9996

        # 一半给中文，一半给英文
        zh_budget = available // 2               # 例如 9996//2 = 4998
        en_budget = available - zh_budget        # 例如 9996-4998 = 4998

        # ── 步骤4：取高频中文字符 ──
        # .most_common(n) 返回频率最高的n个 (element, count) 元组
        # 例如 [("的", 15234), ("一", 8234), ("是", 7123), ...]
        zh_top = zh_counter.most_common(zh_budget)

        # 从ID=4开始分配（0-3已被特殊token占用）
        for i, (char, _) in enumerate(zh_top):
            idx = 4 + i                          # 中文起始ID = 4
            self.stoi[char] = idx                # stoi["的"] = 4
            self.itos[idx] = char                # itos[4] = "的"

        # ── 步骤5：计算英文token起始ID ──
        # 中文占用了多少个字符
        self.zh_offset = len(zh_top)              # 例如 4998
        # 英文从中文结束的下一个位置开始
        self.en_start_id = 4 + self.zh_offset     # 例如 4+4998 = 5002

        # ── 步骤6：取高频英文单词 ──
        en_top = en_counter.most_common(en_budget)
        for i, (word, _) in enumerate(en_top):
            idx = self.en_start_id + i            # 例如 5002+i
            self.stoi[word] = idx                 # stoi["the"] = 5002
            self.itos[idx] = word                 # itos[5002] = "the"

        # ── 打印统计 ──
        print(f"  [Tokenizer] 中文词表: {len(zh_top)} 个字符")
        print(f"  [Tokenizer] 英文词表: {len(en_top)} 个单词")
        print(f"  [Tokenizer] 联合词表总计: {len(self.stoi)} 个 token")
        # 注意：len(stoi) = 4(特殊) + zh_top + en_top ≤ vocab_size

    # ==========================================================================
    # 编码方法：把文本转成整数列表
    # ==========================================================================

    def encode_zh(self, text: str, add_special: bool = True) -> List[int]:
        """
        【中文编码】：中文句子 → token ID 序列

        【编码流程】
          1. tokenize_zh(text) → 拆成字符列表，如 ["你", "好"]
          2. 查字典 stoi，找不到的用 UNK_IDX(3) 代替
          3. （可选）头尾加 SOS/EOS

        【为什么要加 SOS/EOS】
          - SOS: 告诉解码器"开始生成"。decoder的第一个输入永远是SOS。
          - EOS: 告诉解码器"生成结束"。推理时遇到EOS就停止。
          训练时不做推理，但保持格式一致很重要。

        【输入输出示例】
          encode_zh("你好")         → [1, ID(你), ID(好), 2]
          encode_zh("你好", False)  → [ID(你), ID(好)]
          encode_zh("𠀀")           → [1, 3, 2]   # 生僻字 → UNK

        【参数说明】
          text: 原始中文字符串
          add_special: 是否在头尾加SOS/EOS，默认True（训练和推理都需要）
        """
        # 1. 分词：按字符拆开
        tokens = self.tokenize_zh(text)

        # 2. 查字典：stoi.get(token, default)
        #    如果token在词表中 → 返回对应ID
        #    如果token不在词表中 → 返回UNK_IDX (3)
        ids = [self.stoi.get(t, self.UNK_IDX) for t in tokens]

        # 3. 加首尾标记（如果需要）
        if add_special:
            # 在开头插入 SOS(1)，在结尾追加 EOS(2)
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]

        return ids

    def encode_en(self, text: str, add_special: bool = True) -> List[int]:
        """
        【英文编码】：英文句子 → token ID 序列

        和 encode_zh 完全相同的逻辑，只是调用的分词函数不同。
        tokenize_en 按单词+标点切分，tokenize_zh 按字符切分。

        【Teacher Forcing 中的用法】
          编码完整的目标句子：[SOS, word1, word2, ..., wordN, EOS]
          然后 tgt_input  = 去掉最后的EOS → [SOS, word1, ..., wordN]
               tgt_output = 去掉最前的SOS → [word1, word2, ..., EOS]
          这样模型学会：看到 [SOS, word1] 预测 word2，以此类推。

        【输入输出示例】
          encode_en("hello world")        → [1, ID(hello), ID(world), 2]
          encode_en("hello world", False) → [ID(hello), ID(world)]
        """
        # 1. 分词：按单词+标点拆开
        tokens = self.tokenize_en(text)

        # 2. 查字典
        ids = [self.stoi.get(t, self.UNK_IDX) for t in tokens]

        # 3. 加首尾标记
        if add_special:
            ids = [self.SOS_IDX] + ids + [self.EOS_IDX]

        return ids

    # ==========================================================================
    # 解码方法：把整数列表还原成文本
    # ==========================================================================

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        【英文解码】：token ID 序列 → 英文文本

        【解码流程】
          1. 遍历ID列表
          2. 跳过特殊token (PAD/SOS/EOS/UNK)
          3. 用 itos 字典把ID转为单词字符串
          4. 用空格拼接所有单词
          5. 修正标点前的多余空格（英文书写规范）

        【为什么只解码英文】
          训练方向是 中文→英文，所以推理输出永远是英文。
          如果要做英文→中文，需要这个方法改成逐字拼接（无空格）。

        【标点修正详解】
          "hello , world !"  → 修正 → "hello, world!"
          正则 r" ([.,!?;:)'\"])" 匹配 "空格+标点"，删掉空格
          正则 r"([(]) " 匹配 "左括号+空格"，删掉空格

        【输入输出示例】
          decode([1, 18, 20, 2])    → "hello world"  (跳过了SOS/EOS)
          decode([1, 18, 0, 0, 0])  → "hello"        (跳过了PAD)
          decode([1, 18, 3, 2])     → "hello"        (跳过了UNK)
        """
        # 跳过这些特殊token
        specials = {self.PAD_IDX, self.SOS_IDX, self.EOS_IDX, self.UNK_IDX}

        tokens = []
        for idx in ids:
            # 1. 跳过特殊token
            if skip_special and idx in specials:
                continue

            # 2. ID → 单词
            #    itos.get(idx, default) 如果idx不在词表中，返回UNK_TOKEN
            token = self.itos.get(idx, self.UNK_TOKEN)

            # 3. 如果是UNK，跳过（不输出"<unk>"这种占位符）
            if token != self.UNK_TOKEN:
                tokens.append(token)

        # 4. 用单个空格连接所有token
        #    例如 ["hello", ",", "world", "!"] → "hello , world !"
        text = " ".join(tokens)

        # 5. 修正英文标点格式：删掉标点前的空格
        #    "hello , world !" → "hello, world!"
        #    解释：r" ([.,!?;:)'\"])" 匹配 "空格 + 标点"
        #          r"\1" 只保留标点（去掉前面的空格）
        text = re.sub(r" ([.,!?;:)'\"])", r"\1", text)

        # 6. 修正左括号格式：删掉左括号后面的空格
        #    "( hello )" → "(hello)"
        text = re.sub(r"([(]) ", r"\1", text)

        return text

    # ==========================================================================
    # 批处理：一次编码多条句子，自动padding到相同长度
    # ==========================================================================

    def encode_batch_zh(self, texts: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【批量中文编码】一次处理多条中文句子，自动padding。

        【为什么需要批处理】
          GPU 高效计算要求所有输入形状一致。
          但自然语言句子长度各不相同（"你好"2字 vs "深度学习是..."10字）。
          所以需要把所有句子补到相同长度（batch内最长，或全局max_len）。

        【输出格式】
          padded: (batch_size, max_len)  整数张量，短句尾部填0 (PAD)
          mask:   (batch_size, max_len)  布尔张量，True=有效token，False=PAD

        【参数说明】
          texts: 中文句子列表 ["你好", "世界", "深度学习"]
          max_len: 最大序列长度，超过的截断，不足的补PAD

        【返回值说明】
          Tuple[torch.Tensor, torch.Tensor]:
            - 第一个张量 shape=(B, max_len)，dtype=torch.long，内容为token IDs
            - 第二个张量 shape=(B, max_len)，dtype=torch.bool，True=有效位置

        【使用示例】
          texts = ["你好", "今天天气很好"]
          padded, mask = tokenizer.encode_batch_zh(texts, max_len=8)
          # padded = [[1, ID(你), ID(好), 2, 0, 0, 0, 0],
          #           [1, ID(今), ID(天), ID(天), ID(气), ID(很), ID(好), 2]]
          # mask   = [[T, T,     T,     T, F, F, F, F],
          #           [T, T,     T,     T, T,     T,     T,     T]]
        """
        batch_ids = []
        for text in texts:
            # 逐条编码（含SOS/EOS）
            ids = self.encode_zh(text, add_special=True)
            # 截断到 max_len（超过的部分丢弃）
            ids = ids[:max_len]
            batch_ids.append(ids)

        # 统一padding
        return self._pad_batch(batch_ids, max_len)

    def encode_batch_en(self, texts: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        【批量英文编码】逻辑完全同 encode_batch_zh，只是调用 encode_en。

        这个函数主要用于：
          1. 训练时批量编码目标句子（用于Teacher Forcing）
          2. 验证时批量计算loss
        """
        batch_ids = []
        for text in texts:
            ids = self.encode_en(text, add_special=True)
            ids = ids[:max_len]
            batch_ids.append(ids)

        return self._pad_batch(batch_ids, max_len)

    @staticmethod
    def _pad_batch(batch_ids: List[List[int]], max_len: int):
        """
        【核心填充函数】把不等长的ID序列填充为整齐的矩阵。

        【填充策略】
          - 用 PAD_IDX (0) 填充空白位置
          - 用布尔 mask 标记哪些位置是有效token (True) 哪些是padding (False)
          - 这个 mask 在 Attention 层被转为 -inf，让模型忽略padding位置

        【为什么用 PAD=0】
          1. 方便：torch.full(fill_value=0) 天然支持
          2. 安全：PAD 不会被误认为有效token（0不在有效ID范围内）
          3. 高效：Embedding 的 padding_idx=0 可以让PAD的梯度始终为0

        【mask 的作用链路】
          mask (B,S) → model.create_padding_mask → (B,1,1,S) with -inf
          → 加到 attention scores 上 → softmax后padding位置权重≈0
          → 模型完全忽略padding位置的任何信息

        【参数说明】
          batch_ids: [[1,7,4,2], [1,18,2], ...] 不等长ID列表
          max_len: 目标统一长度

        【返回值】
          padded: (B, max_len) long tensor，填充了0的位置
          mask: (B, max_len) bool tensor，True=有效 False=padding
        """
        batch_size = len(batch_ids)

        # torch.full: 创建指定形状的张量，全部填充为 fill_value
        # 这里创建 (batch_size, max_len) 的全0张量（0 = PAD_IDX）
        padded = torch.full(
            (batch_size, max_len),
            fill_value=0,          # PAD_IDX
            dtype=torch.long       # 整数类型（token ID 用 long）
        )

        # torch.zeros: 创建 (batch_size, max_len) 的全False张量
        # 表示初始时所有位置都是"无效的"（等待后面逐一设为True）
        mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool       # 布尔类型
        )

        # 逐条填入
        for i, ids in enumerate(batch_ids):
            # 取实际长度（不能超过max_len）
            length = min(len(ids), max_len)

            # 把这条句子的token ID复制到 padded 的第i行前length列
            # 后面的列保持0（PAD）
            padded[i, :length] = torch.tensor(
                ids[:length],
                dtype=torch.long
            )

            # 把 mask 的前length列设为True，表示这些位置是有效token
            mask[i, :length] = True

        return padded, mask

    # ==========================================================================
    # 保存与加载：词表持久化
    # ==========================================================================

    def save(self, path: str):
        """
        【保存词表到JSON文件】

        为什么保存为JSON而不是pickle？
          - JSON是人类可读的，你可以直接打开 tokenizer.json 查看每个token的ID
          - JSON跨版本兼容，不会因为Python版本变化而无法加载
          - JSON可被任何语言读取（JS、Java、Rust等），便于部署

        保存的内容：
          - vocab_size: 词表大小
          - stoi: token→ID 字典（约10000个条目）
          - zh_offset / en_start_id: 英文token起始位置

        注意：不保存 itos，因为可以从 stoi 反推（v→k）。
        """
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "zh_offset": self.zh_offset,
            "en_start_id": self.en_start_id,
        }

        # json.dump: Python对象 → JSON文件
        # ensure_ascii=False: 保留中文字符，不转义为\uXXXX
        # indent=2: 格式化缩进2空格，方便人类阅读
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[Tokenizer] 词表已保存到 {path}")

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        """
        【从JSON文件加载词表】

        使用 @classmethod 是因为需要先创建一个空的 tokenizer 实例，
        然后把JSON中的数据填进去。

        加载后 tokenizer 的状态和训练时完全一致，
        可以直接用于推理、继续训练等。

        注意：JSON的key只能是字符串，所以 stoi 里的数字key会被转成字符串。
        但我们的key本来就是字符串（token文本），所以不受影响。
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 用JSON中的vocab_size创建空tokenizer
        tokenizer = cls(vocab_size=data["vocab_size"])

        # 恢复 stoi 字典
        tokenizer.stoi = data["stoi"]

        # 恢复 itos（反向字典）
        # JSON中可能存储了itos（新版），也可能需要从stoi反推（旧版）
        if "itos" in data:
            # JSON的key只能是字符串，但我们需要int key
            tokenizer.itos = {int(k): v for k, v in data["itos"].items()}
        else:
            # 从 stoi 反推：{v: k for k, v in stoi.items()}
            # 但 stoi 的 value 可能是 int 或 str（取决于JSON序列化方式）
            tokenizer.itos = {
                int(v) if isinstance(v, str) else v: k
                for k, v in data["stoi"].items()
            }

        # 恢复边界信息
        tokenizer.zh_offset = data.get("zh_offset")
        tokenizer.en_start_id = data.get("en_start_id")

        # 安全兜底：如果itos缺失或不完整，从stoi重建
        if not tokenizer.itos or len(tokenizer.itos) < len(tokenizer.stoi):
            tokenizer.itos = {
                int(v) if isinstance(v, str) else v: k
                for k, v in tokenizer.stoi.items()
            }

        print(f"[Tokenizer] 词表已加载: {len(tokenizer.stoi)} 个 token")
        return tokenizer

    def __len__(self):
        """返回词表大小，方便用 len(tokenizer) 获取"""
        return len(self.stoi)
