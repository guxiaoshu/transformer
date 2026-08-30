# transformer
手撸transformer

使用方法：
https://huggingface.co/datasets/Helsinki-NLP/opus-100
下载训练集


打开config.py
把data_dir改成训练集的地址
data_dir: str = "D:/Codefield/Transformer/huggingface/datasets/Helsinki-NLP___opus-100"

这个D:/Codefield/Transformer/huggingface/datasets/Helsinki-NLP___opus-100是我的地址，要换掉


—————————— edu 版 ——————————

训练：

powershell
cd D:\Codefield\Transformer\transformer\edu
python main.py train

D:\Codefield\Transformer\transformer\edu   改成自己的地址

训练会自动完成：
读入并清洗 50000 条中英句对
现场构建词表，保存到 `checkpoints/tokenizer.json`不用提前训练分词器
开始训练（30 个 epoch），每 50 个 batch 打印一次日志
验证集 loss 创新低时保存 `checkpoints/best_model.pt`

翻译：
powershell
# 输入一句翻译一句的对话模式
python main.py infer

对话模式下输入 `exit`（或 `quit` / `q`）退出；在句子后加 `--verbose` 可以看到内部 token、各层张量形状等调试信息。


# 单句翻译
python main.py infer "你好世界"


# 子命令
python main.py train   开始训练 
python main.py train --resume  从 `best_model.pt` 从上次的进度继续训练（换学习率、继续跑时用）
python main.py infer    交互式翻译
python main.py infer "XXX"    翻译单句 
python main.py test   自测：分词器 → 模型构建 → 前向传播 → Mask → 损失 → 推理，用来确认环境没坏 ，刚下载好代码、或改了 `model.py` / `tokenizer.py` 之后，先跑一次 `python main.py test`，如果 6 个小测试全部正常打印出结果，说明环境 OK，可以放心训练。





—————————— pro版 ——————————
环境依赖多一个：
pip install torch pyarrow sentencepiece

训练：

要先训练 BPE 分词器
```powershell
cd D:\Codefield\Transformer\transformer\pro
python train_tokenizer.py
```
D:\Codefield\Transformer\transformer\pro   改成自己的地址

分词器会产生：
`bpe_corpus.txt` 训练分词器时的训练语料（中间产物，用完可删）
`spm.model`  BPE 模型（模型真正要加载的文件）
`spm.vocab` 给人看的，实际编码用不到
3个文件



再训练模型：
```powershell
python main.py train
```

翻译：
```powershell
# 对话模式
python main.py infer

# 单句翻译
python main.py infer "你好世界"
```

# 子命令
python train_tokenizer.py  训练 BPE 分词器（先跑）
python main.py train  训练模型 
python main.py train --resume  从 `best_model.pt` 恢复  
python main.py infer`  对话式翻译  
python main.py infer "XXX"  翻译单句 
python main.py test`   自测各模块（没训 BPE 时自动跳过分词/推理测试） 



# 如果OOM：
优先把 `config.py` 里的 `batch_size` 调小，比如 64 → 32 → 16。
其他bug问Claude大人