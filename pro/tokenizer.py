
from typing import List, Optional


class BpeTokenizer:


    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<sos>"   # 对应 sentencepiece 的 <s>
    EOS_TOKEN = "<eos>"   # 对应 sentencepiece 的 </s>
    UNK_TOKEN = "<unk>"

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    def __init__(self, model_path: Optional[str] = None):
        self.sp = None
        self.model_path = model_path
        if model_path is not None:
            self.load(model_path)

    def load(self, path: str) -> "BpeTokenizer":
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.model_path = path
        return self

    @classmethod
    def from_file(cls, path: str) -> "BpeTokenizer":
        return cls(path)

    def _encode(self, text: str, add_special: bool) -> List[int]:
        if add_special:
            # 训练时设置了 bos_id=1, eos_id=2，encode 默认加 <s> 和 </s>
            return self.sp.encode(text, out_type=int)
        # 不加特殊标记：兼容新旧版 sentencepiece API
        try:
            return self.sp.Encode(text, out_type=int)   # 新版
        except AttributeError:
            return self.sp.encode(text, out_type=int,
                                  add_bos=False, add_eos=False)  # 旧版

    def encode_zh(self, text: str, add_special: bool = True) -> List[int]:
        # 中英文共用同一个 BPE 模型，所以 encode_zh 和 encode_en 逻辑相同
        return self._encode(text, add_special)

    def encode_en(self, text: str, add_special: bool = True) -> List[int]:
        return self._encode(text, add_special)

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids
                   if i not in (self.PAD_IDX, self.SOS_IDX,
                                self.EOS_IDX, self.UNK_IDX)]
        if not ids:
            return ""
        # sentencepiece 的 decode 会自动把子词拼回文本，英文空格由 ▁ 还原
        return self.sp.decode(ids)

    def id_to_piece(self, idx: int) -> str:
        try:
            return self.sp.id_to_piece(idx)
        except Exception:
            return self.UNK_TOKEN

    def __getitem__(self, idx: int) -> str:
        return self.id_to_piece(idx)

    def __len__(self) -> int:
        return self.sp.get_piece_size()
