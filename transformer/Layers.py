''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    ''' 这里的的forward调用了MultiHeadAttention#forward方法，
        其中入参k、q、v的维度是：(batch_size, seq_length, d_model)，三维tensor，输入数据里没有n_head

        经过attention操作之后的O，维度和输入是一样的，在公式里是N*d,在这里是：(batch_size, seq_length, d_model)。
        
    '''
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    ''' 对于解码器，需要注意的是每个层中有两个多头注意力处理单元，其中第一个多头注意力处理单元，接收的是解码器自己的输出作为输入。
        而第二个多头注意力单元，query（q）接收的是解码器第一个多头注意力处理单元的输出作为输入，意味着需要根据自己已经处理的内容来控制输出。
        k和v接收的是编码器encoder的输出seq_len * d_model作为输入。
        其中需要注意的是k、v的seq_len和第一个多头注意力生成的q的seq_len不一样，因为q中的seq_len是一步步生成新增的。

        注意：这里来核心的问题也是：如果在第二个多头注意力中，q的seq_len和k、v的seq_len不一样，应该如何处理？没有看到。
        在点积Attention里，是可以处理：q的seq_len和k的seq_len不一样的情况的，只要k、v的seq_len一样就可以。最后输出的是seq_len_q * d 的output
        对应了decoder第一个多头注意力的输入。
    '''
    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
