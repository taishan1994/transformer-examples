from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding(nn.Module):
    """
    pos(k, 2i)=sin(k/(10000**(2i/d)))
    pos(k, 2i+1)=cos(k/(10000**(2i/d)))
    :param seq_len: 句子长度
    :param dim_model: 隐含层的维度
    :return:
    """

    def __init__(self, dim_model, seq_len):
        super(PositionalEncoding, self).__init__()
        # 让该参数不更新
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(seq_len, dim_model))

    def _get_sinusoid_encoding_table(self, seq_len, dim_model):
        def get_sinusoid_encoding_table(k):
            # 注意这里为什么i//2
            return [k * np.power(10000, 2 * (i // 2) / dim_model) for i in range(dim_model)]

        sinusoid_table = np.array([get_sinusoid_encoding_table(k) for k in range(seq_len)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).squeeze(0)

    def forward(self, x):
        # print(x.shape)
        # print(self.pos_table.shape)
        # print(self.pos_table[:x.size(1), :].clone().detach().unsqueeze(0).shape)
        return x + self.pos_table[:x.size(1), :].clone().detach().unsqueeze(0)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,  # 输入维度
                 d_inner,  # 输出维度
                 n_head,  # 头数
                 d_k,  # k的维度
                 d_v,  # v的维度
                 dropout=0.3):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(self,
                 n_src_vocab,  # source词汇表大小
                 d_word_vec,  # 词嵌入维度
                 n_layers,  # 层数
                 n_head,  # 头数
                 d_k,  # k的维度
                 d_v,  # v的维度
                 d_model,  # 编码器输入维度
                 d_inner,  # 编码器输出维度
                 pad_idx,  # padding的id
                 dropout=0.3,  # dropout
                 seq_len=512,  # 句子长度
                 scale_emb=False  # 注意力是否缩放
                 ):
        super(Encoder, self).__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output, []


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        # 除了解码器的自注意力之外，还有编码-解码注意力
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, seq_len=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, seq_len=seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).long().unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class TransformerSeq2Seq(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, src_seq_len=200,
            trg_seq_len=200, trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, seq_len=src_seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, seq_len=trg_seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
            'To facilitate the residual connections, \
             the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        # 这里的注意力掩码很重要
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5
        # print(seq_logit.shape)
        # return seq_logit.view(-1, seq_logit.size(2))
        return seq_logit


class TransformerEncoder(nn.Module):
    def __init__(self, n_src_vocab, src_pad_idx,
                 d_word_vec=512, d_model=512, d_inner=2048,
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, src_seq_len=200,
                 label_num=None):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, seq_len=src_seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=False)
        self.fc = nn.Linear(d_model, label_num)

    def forward(self, src_seq):
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        enc_output, *_ = self.encoder(src_seq, src_mask)
        enc_output = torch.sum(enc_output, 1).squeeze()
        logits = self.fc(enc_output)
        return logits


class TransformerDecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.dec_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, dec_enc_attn_mask=None):
        dec_output, dec_enc_attn = self.dec_attn(
            dec_input, dec_input, dec_input, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_enc_attn


class TransformerDecoder(nn.Module):
    def __init__(self, n_trg_vocab, trg_pad_idx, d_word_vec, d_model, d_inner,
                 n_layers, n_head, d_k, d_v, dropout=0.1, trg_seq_len=256, scale_emb=False):
        super(TransformerDecoder, self).__init__()
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=trg_pad_idx)
        self.trg_pad_idx = trg_pad_idx
        self.position_enc = PositionalEncoding(d_word_vec, seq_len=trg_seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            TransformerDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.fc = nn.Linear(d_model, n_trg_vocab)

    def forward(self, trg_seq, return_attns=False):
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        # dec_enc_attn_list = []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_enc_attn = dec_layer(
                dec_output, dec_enc_attn_mask=trg_mask)
            # dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        # if return_attns:
        #     return dec_output, dec_enc_attn_list
        logits = self.fc(dec_output)
        return logits


if __name__ == '__main__':
    src_seq = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0],
        [1, 2, 3, 4, 5, 6, 0],
        [1, 3, 4, 0, 0, 0, 0],
        [1, 3, 4, 1, 0, 0, 0],
    ])
    trg_seq = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 0, 0],
        [1, 3, 4, 0, 0, 0, 0, 0],
        [1, 2, 0, 0, 0, 0, 0, 0],
    ])
    src_mask = get_pad_mask(src_seq, 0)
    pprint(src_mask)
    pprint(trg_seq)
    trg_mask = get_pad_mask(trg_seq, 0) & get_subsequent_mask(trg_seq)
    pprint(trg_mask)

    transformer = TransformerSeq2Seq(
        n_src_vocab=50000,
        n_trg_vocab=50000,
        src_pad_idx=0,
        trg_pad_idx=0,
        d_word_vec=768,
        d_model=768,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        src_seq_len=256,
        trg_seq_len=256,
        trg_emb_prj_weight_sharing=False,
        emb_src_trg_weight_sharing=False,
    )

    # print(transformer)
    #
    # for name, _ in transformer.named_parameters():
    #     print(name)

    src_seq = torch.ones((2, 256)).long()
    trg_seq = torch.ones((2, 256)).long()
    output = transformer(src_seq, trg_seq)
    print(output.shape)

    cls = TransformerEncoder(
        n_src_vocab=50000,
        src_pad_idx=0,
        d_word_vec=768,
        d_model=768,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        src_seq_len=256,
        label_num=10,
    )

    print(cls(src_seq).shape)

    gen = TransformerDecoder(
        n_trg_vocab=50000,
        trg_pad_idx=0,
        d_word_vec=768,
        d_model=768,
        d_inner=2048,
        n_layers=6,
        n_head=8,
        d_k=64,
        d_v=64,
        dropout=0.1,
        trg_seq_len=256,
    )

    print(gen(trg_seq).shape)
