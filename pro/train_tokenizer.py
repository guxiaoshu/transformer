


import os

import pyarrow.ipc as ipc
import sentencepiece as spm

from config import config
from tokenizer import BpeTokenizer


def load_sentences(data_dir: str, max_pairs: int = 500_000):
# 从 OPUS-100 顺序读前 max_pairs 个句对，返回中英文混合句子列表。

#     OPUS-100 的 arrow 文件顺序本身是打散的，顺序读即可，无需随机抽样。

    arrow_dir = os.path.join(data_dir, "en-zh", "0.0.0")
    subdirs = [d for d in os.listdir(arrow_dir)
               if os.path.isdir(os.path.join(arrow_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"未找到 Arrow 目录: {arrow_dir}")
    arrow_dir = os.path.join(arrow_dir, subdirs[0])
    train_file = os.path.join(arrow_dir, "opus-100-train.arrow")

    sentences = []
    pairs = 0
    with ipc.open_stream(train_file) as reader:
        for batch in reader:
            translations = batch.column("translation").to_pylist()
            for row in translations:
                sentences.append(row["zh"])
                sentences.append(row["en"])
                pairs += 1
                if pairs >= max_pairs:
                    break
            if pairs >= max_pairs:
                break

    print(f"  读取 {pairs} 个句对，共 {len(sentences)} 句（中英混合）")
    return sentences


def main():
    print("=" * 60)
    print("  先训练 BPE 分词器")
    print("=" * 60)

    sentences = load_sentences(config.data_dir, max_pairs=500_000)

    # 写入语料文件
    os.makedirs(config.save_dir, exist_ok=True)
    corpus_path = os.path.join(config.save_dir, "bpe_corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")
    print(f"  语料写入 {corpus_path}")

    # 训练 BPE 模型
    # model_prefix=.../spm 会生成 spm.model 和 spm.vocab
    model_prefix = os.path.join(config.save_dir, "spm")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=config.vocab_size,      # 32000
        model_type="bpe",                  # BPE 子词
        character_coverage=0.9995,         # 中文字符多，需要高覆盖率
        # ── 固定特殊 token id，和 config / 模型 / data.py 的约定一致 ──
        pad_id=0,                          # <pad> = 0
        bos_id=1,                          # <s>   = 1（相当于 <sos>）
        eos_id=2,                          # </s>  = 2（相当于 <eos>）
        unk_id=3,                          # <unk> = 3
        max_sentence_length=1024,
        num_threads=8,
        hard_vocab_limit=False,            # 语料不足时允许词表略小于 32000，不报错
    )

    print(f"  BPE 模型训练完成 → {config.tokenizer_path}")

    # 验证
    tok = BpeTokenizer.from_file(config.tokenizer_path)
    print(f"  实际词表大小: {len(tok)}")
    demo_zh = tok.encode_zh("你好世界，机器学习很有趣")
    demo_en = tok.encode_en("hello world, machine learning is fun")
    print(f"  中文 '你好世界，机器学习很有趣' → {demo_zh}")
    print(f"     decode 还原 → {tok.decode(demo_zh)!r}")
    print(f"  英文 'hello world, machine learning is fun' → {demo_en}")
    print(f"     decode 还原 → {tok.decode(demo_en)!r}")


if __name__ == "__main__":
    main()
