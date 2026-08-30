
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class PositionalEncoding(nn.Module):


    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):

        super().__init__()

        # Dropout随机关闭一部分神经元，防止过拟合 

        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len, d_model)  # 先全填0


        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)


        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *    # 0, 2, 4, ..., d_model-2
            (-math.log(10000.0) / d_model)           # -log(10000)/d_model
        )

        # 填充 pe 

        pe[:, 0::2] = torch.sin(position * div_term)   # (max_len, d_model/2)

        pe[:, 1::2] = torch.cos(position * div_term)   # (max_len, d_model/2)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:


        seq_len = x.size(1)
        out = x + self.pe[:, :seq_len, :]

        # if verbose:
        #     print(f" 输入: {x.shape} → 加位置编码 → 输出: {out.shape}")

        return self.dropout(out)




class MultiHeadAttention(nn.Module):


    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):

        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads   # 每个头的维度

        #  四个线性投影矩阵 
        # 训练中不断更新
        self.W_q = nn.Linear(d_model, d_model, bias=False)

        self.W_k = nn.Linear(d_model, d_model, bias=False)

        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        #  存储最近一次计算的注意力权重
        self.attn_weights = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:

        B, seq_len, _ = x.shape
        x = x.view(B, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)   # → (B, n_heads, seq_len, d_k)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:

        B, _, seq_len, _ = x.shape
        x = x.transpose(1, 2)              # → (B, seq_len, n_heads, d_k)
        x = x.contiguous()                 # → 内存连续化
        return x.view(B, seq_len, self.d_model)   # → (B, seq_len, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor = None, kv_cache: dict = None,
                verbose: bool = False) -> torch.Tensor:

        B = query.size(0)

        if verbose:
            k_shape = key.shape if key is not None else "cached"
            v_shape = value.shape if value is not None else "cached"
            cache_info = " [KV Cache]" if kv_cache is not None else ""
            print(f"  [MultiHeadAttn] Q:{query.shape} K:{k_shape} V:{v_shape}{cache_info}")

        # 线性投影 → 拆头 
        # Q 总是需要完整计算,永远都是新的
        Q = self._split_heads(self.W_q(query))    # (B, H, Q_len, d_k)

        # 根据是否使用 KV Cache 
        if kv_cache is not None and key is not None:
            # 解码器+ KV Cache
            # 只计算新 token 的 K
            # 然后追加到缓存中和历史 token 拼接。
            # 这避免了重算前面的 K
            K_new = self._split_heads(self.W_k(key))      # (B, H, 1, d_k)
            V_new = self._split_heads(self.W_v(value))    # (B, H, 1, d_k)

            if kv_cache["k"] is not None:
                # 缓存中已有历史 token沿序列维度拼接
                K = torch.cat([kv_cache["k"], K_new], dim=2)  # (B, H, t+1, d_k)
                V = torch.cat([kv_cache["v"], V_new], dim=2)
            else:
                # 第一次调用，缓存为空 ，直接使用新 K/V
                K, V = K_new, V_new

            # 原地更新缓存 
            kv_cache["k"] = K
            kv_cache["v"] = V

        elif kv_cache is not None and key is None:
            # 解码器cross attention+ KV Cache
            # 编码器输出在推理时不变 
            # 已预计算并存入缓存。直接使用，不追加。
            K = kv_cache["k"]   # (B, H, S, d_k)  S=源语言长度，固定不变
            V = kv_cache["v"]   # (B, H, S, d_k)

        else:
            # 完整计算，废弃掉算了
            K = self._split_heads(self.W_k(key))      # (B, H, K_len, d_k)
            V = self._split_heads(self.W_v(value))    # (B, H, V_len, d_k)


        # 计算注意力分数

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # mask

        if mask is not None:
            scores = scores + mask

        # Softmax 归一化 

        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        self.attn_weights = attn_weights.detach()

        # 加权求和
        context = torch.matmul(attn_weights, V)



        #合并头
        output = self.W_o(self._combine_heads(context))

        if verbose:
            print(f"输出 {output.shape}")#要打印出来看变化

        return output



class FeedForward(nn.Module):


    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):

        super().__init__()


        # 先升维在高维空间做非线性变换再降维回来
        self.linear1 = nn.Linear(d_model, d_ff)


        # 降维回原始维度
        self.linear2 = nn.Linear(d_ff, d_model)

        # ReLU 之后的 dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:

        # 升维 + 非线性
        out = self.linear2(
            self.dropout(
                F.relu(
                    self.linear1(x)
                )
            )
        )


        return out


class EncoderLayer(nn.Module):


    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):

        super().__init__()

        # 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None,
                verbose: bool = False) -> torch.Tensor:

        # ── Post-LN：先做子层 + 残差相加，再做 LayerNorm（改回原论文顺序）──
        attn_out = self.self_attn(x, x, x, mask=src_mask, verbose=verbose)
        x = self.norm1(x + self.dropout(attn_out))

        x = self.norm2(x + self.dropout(self.ffn(x, verbose=verbose)))

        if verbose:
            print(f" Encoder的输出 {x.shape}")

        return x



class DecoderLayer(nn.Module):


    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # 掩码自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 交叉注意力Q来自解码器, KV来自编码器
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # 前馈网络
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Layer ×3
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                tgt_mask: torch.Tensor = None, src_mask: torch.Tensor = None,
                verbose: bool = False,
                self_kv_cache: dict = None, cross_kv_cache: dict = None
                ) -> torch.Tensor:

        # ── Post-LN：每个子层先残差相加，再做 LayerNorm（改回原论文顺序）──
        attn_out = self.self_attn(
            x, x, x,
            mask=tgt_mask, kv_cache=self_kv_cache, verbose=verbose
        )
        x = self.norm1(x + self.dropout(attn_out))

        if cross_kv_cache is not None:
            cross_out = self.cross_attn(
                x, None, None,
                mask=src_mask, kv_cache=cross_kv_cache, verbose=verbose
            )
        else:
            cross_out = self.cross_attn(
                x, enc_out, enc_out,
                mask=src_mask, verbose=verbose
            )
        x = self.norm2(x + self.dropout(cross_out))

        x = self.norm3(x + self.dropout(self.ffn(x, verbose=verbose)))

        if verbose:
            print(f"  Decoder的输出: {x.shape}")

        return x


class Seq2SeqTransformer(nn.Module):


    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_encoder_layers: int, n_decoder_layers: int, d_ff: int,
                 dropout: float = 0.1, max_len: int = 64, pad_idx: int = 0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model

        #共享词嵌入矩阵

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # 缩放因子
        self.embed_scale = math.sqrt(d_model)   # 例如 √256 = 16

        # 位置编码 
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # 编码器
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # 输出投影头
        # 把 d_model 维的特征映射到 logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        # 初始化
        self._init_parameters()

        # 打印信息


    def _init_parameters(self):


          # Xavier 初始化
        for p in self.parameters():
            if p.dim() > 1:

                nn.init.xavier_uniform_(p)

    @staticmethod
    def create_padding_mask(mask: torch.Tensor) -> torch.Tensor:
 
        return mask.unsqueeze(1).unsqueeze(2).float().log()

    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def create_tgt_mask(tgt_padding_mask: torch.Tensor) -> torch.Tensor:

        B, T = tgt_padding_mask.shape
        device = tgt_padding_mask.device

        # 因果mask
        causal = Seq2SeqTransformer.create_causal_mask(T, device)  # (1, 1, T, T)

        # Padding mask: (B, 1, 1, T)
        pad = tgt_padding_mask.unsqueeze(1).unsqueeze(2).float().log()

        # 广播加法(1,1,T,T) + (B,1,1,T) → (B,1,T,T)广播真是个好东西
        return causal + pad



    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None,
                verbose: bool = False) -> torch.Tensor:

        if verbose:
            print("=" * 60)
            print("  前向传播 ")
            print("=" * 60)
            print(f"  src: {src.shape} ")
            print(f"  tgt_input: {tgt_input.shape} ")

        # 创建 Attention Mask
        # 把布尔mask转为带 -inf 的浮点mask
        src_attn_mask = self.create_padding_mask(src_mask) \
            if src_mask is not None else None
        tgt_attn_mask = self.create_tgt_mask(tgt_mask) \
            if tgt_mask is not None else None


        # 词嵌入 + 缩放 
        # 为什么不直接用 embedding
        # 因为 embedding 初始值很小
        # 而 positional encoding 在 [-1,1] 范围。
        # 乘以 √d_model 放大了embedding，让两者在相同量级。
        src_emb = self.embedding(src) * self.embed_scale  # (B, S, D)
        tgt_emb = self.embedding(tgt_input) * self.embed_scale  # (B, T, D)



        # 编码器
        x = src_emb
        x = self.pos_encoding(x)   # 注入位置信息
        for i, layer in enumerate(self.encoder_layers):
            if verbose:
                print(f"\n[Encoder Layer {i+1}/{len(self.encoder_layers)}]")
            x = layer(x, src_mask=src_attn_mask, verbose=verbose)
        enc_out = x   # (B, S, D)



        # 解码器 
        x = tgt_emb
        x = self.pos_encoding(x)
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x, enc_out,
                      tgt_mask=tgt_attn_mask, src_mask=src_attn_mask,
                      verbose=verbose)
        dec_out = x


        # 输出投影
        logits = self.output_proj(dec_out)  # (B, T, V)

        return logits


    def encode_for_inference(self, src: torch.Tensor, src_mask: torch.Tensor,
                             verbose: bool = False,
                             return_caches: bool = False) -> torch.Tensor:

        src_emb = self.embedding(src) * self.embed_scale
        src_emb = self.pos_encoding(src_emb)
        src_attn_mask = self.create_padding_mask(src_mask)

        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_mask=src_attn_mask)

        enc_out = src_emb

        if return_caches:
            # 预计算所有解码器层的交叉注意力KV
            # 为什么可以预计算？交叉注意力的 Q 来自解码器，随时变化
            # 但 K 和 V 来自编码器输出（整个推理过程不变）。
            # 预计算后每个 decode step 直接用，避免重复的 W_k和W_v 投影。
            cross_kv_caches = []
            for layer in self.decoder_layers:
                cache = {
                    "k": layer.cross_attn._split_heads(
                        layer.cross_attn.W_k(enc_out)
                    ),  # (1, H, S, d_k)
                    "v": layer.cross_attn._split_heads(
                        layer.cross_attn.W_v(enc_out)
                    ),  # (1, H, S, d_k)
                }
                cross_kv_caches.append(cache)
            return enc_out, cross_kv_caches

        return enc_out

    def decode_step(self, tgt_token: torch.Tensor, enc_out: torch.Tensor,
                    tgt_mask: torch.Tensor, src_mask: torch.Tensor,
                    past_len: int = 0, verbose: bool = False,
                    self_kv_caches: list = None,
                    cross_kv_caches: list = None) -> torch.Tensor:

        # if self_kv_caches is not None:
            # KV Cache ，只处理新 token

            # 嵌入新 token（只有 1 个）
            tgt_emb = self.embedding(tgt_token) * self.embed_scale  # [1, 1, D]


            tgt_emb = tgt_emb + self.pos_encoding.pe[:, past_len:past_len+1, :]
            tgt_emb = self.pos_encoding.dropout(tgt_emb)

            tgt_attn_mask = None
            src_attn_mask = self.create_padding_mask(src_mask.expand(1, -1))

            for i, layer in enumerate(self.decoder_layers):
                tgt_emb = layer(
                    tgt_emb, enc_out,
                    tgt_mask=tgt_attn_mask,
                    src_mask=src_attn_mask,
                    verbose=verbose,
                    self_kv_cache=self_kv_caches[i],
                    cross_kv_cache=cross_kv_caches[i] if cross_kv_caches else None,
                )

            return tgt_emb[:, -1:, :]   # [1, 1, D]

        # else:

        #     # 整序列重算，弃用

        #     tgt_emb = self.embedding(tgt_token) * self.embed_scale
        #     tgt_emb = self.pos_encoding(tgt_emb)
        #     tgt_attn_mask = self.create_tgt_mask(tgt_mask)
        #     src_attn_mask = self.create_padding_mask(src_mask.expand(1, -1))

        #     for layer in self.decoder_layers:
        #         tgt_emb = layer(tgt_emb, enc_out,
        #                         tgt_mask=tgt_attn_mask,
        #                         src_mask=src_attn_mask,
        #                         verbose=verbose)
        #     return tgt_emb[:, -1:, :]

    # 获取注意力权重（可视化用）


    def get_attention_weights(self):

        weights = []
        for layer in self.decoder_layers:
            weights.append({
                "self": layer.self_attn.attn_weights,
                "cross": layer.cross_attn.attn_weights,
            })
        return weights
