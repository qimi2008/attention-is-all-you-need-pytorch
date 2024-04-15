''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"

''' 使用unsqueeze（-2）函数调用来向张量添加额外的维度。它在张量的索引-2（倒数第二）处添加了一个单例维度。
    在这种情况下，unsqueeze（-2）用于重塑（seq！=pad_idx）返回的张量，以便它可以用于后续操作，如掩蔽。
    它有助于创建一个与输入张量seq形状相同的掩码张量，但添加了一个额外的维度来表示填充掩码。
    当将掩码应用于另一个张量时，添加这个额外的维度可以帮助广播操作，例如在Transformer模型中的计算过程中屏蔽填充元素。

    seq是一个LongTensor，pad_idx也是也LongTensor，seq != pad_idx的操作，如果两个Tensor的对应位置不相等，
    则对应位置赋值1，否则对应位置赋值0，返回的形状和seq一样。其中pad_idx是数据源中被标记为<blank>的地方的索引。

    多维张量本质上就是一个变换，如果维度是 1 ，那么，1 仅仅起到扩充维度的作用，而没有其他用途，
    因而，squeeze在进行降维操作时，为了加快计算，是可以去掉这些 1 的维度,unsqueeze在进行升维时，也只是增加了一个维度大小是1的维度。

    seq是二维数组(sz_b,len_s),通过unsqueeze(-2)之后，形状变成：(sz_b,1,len_s)
'''
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. 
        从seq.size()的返回值，可以确定seq是二维数组(sz_b,len_s),返回值的形状是：(1,len_s,len_s)
        且mask里的元素的值是下三角的True or False
    '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        ''' 位置编码的公式是：w_k = 10000 ** （-2k/d）,其中-2k/d = -k/(d/2),即取输入维度数量的二分之一，取整。
            在2k的维度上，取：sin(w_k * t),t为seq中的第t个token。
            在2k+1的维度上，取：cos(w_k * t),t为seq中的第t个token
            p_vec = (sin(w_1 * t),cos(w_1 * t),sin(w_2 * t),cos(w_2 * t),sin(w_3 * t),cos(w_3 * t),...,sin(w_d * t),cos(w_d * t)),d为输入向量的维度

           get_position_angle_vec方法返回(1,d_hid)的向量，而sinusoid_table是(n_position,d_hid)的二维向量
        '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i，0::2，其中的2是步长，表示第二的维度的遍历从0开始，每次加2
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 1::2，其中的2是步长，表示第二的维度的遍历从1开始，每次加2

        return torch.FloatTensor(sinusoid_table).unsqueeze(0) # 返回值的形状是：(1,n_position,d_hid)

    ''' 这里在进行位置编码和输入向量编码的按位求和操作
    '''
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        ''' 输入序列变成embedding的操作，是线性变换本身的定义,后续需要把输入灌入src_word_emb，才能获取embedding的编码。
            输入的是：（n_src_vocab,d_word_vec），每个序列中的次元的向量是：(1,d_word_vec)，执行线性变换：W*X+b

            通过一个torch.nn.ModuleList来传入EncoderLayer，数量的定义的参数n_layers.
            nn.ModuleList是构建神经网络的容器，允许我们将多个模块组合成更大的模块
        '''
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx) 
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward，在输入Attention模块之前，需要线性变换+位置编码，同时也需要dropout+layerNor操作。
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
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward 在输入Attention模块之前，需要线性变换+位置编码，同时也需要dropout+layerNor操作。
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        ''' 在DecoderLayer#forward方法里，调用了两次Attention模块，第一个Attention模块的入参是：q=dec_output，k=dec_output，v=dec_output
            第二个attention模块的入参是：q=dec_output，k=enc_output，v=enc_output,其中的dec_output是第一个模块的输出。
        '''
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
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
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
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

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        # 这里核心要注意的就是encoder的enc_output作为decoder的入参
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)  # 最后一个线性转换层Linear，还没有到最后的softmax层。
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))
