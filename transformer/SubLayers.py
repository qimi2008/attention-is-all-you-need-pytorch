''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        ''' nn.Linear创建了神经网络中的线性变换层（全连接层）。输入神经元的数量为d_model，输出神经元的数量为为n_head * d_k，且没有偏差，每个神经元一个维度
            这里其实定义了Y = X*W + b这个线性层。d_model是线性层输入神经元数量。W的值，由torch初始化赋值。后续必须调用self.w_qs(q)把q输入线性层。
            d_model就是输入embedding的维度（线性层的输入）、d_k是线下层的输出维度，这个w_qs，在这里定义的就是X*W这个算子，当前w_qs是一个函数，待
            输入N个d_model维的向量才能执行线性运算活动一个N*d_k的输出，这个输出，在这里就是Q、K、V矩阵。它是通过输入X到线性层，进行X*W操作，获得的输出。
            
        '''
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

        ''' 在Layers类里对forward方法使用，是把enc_input作为该方法对q、k、v的赋值传进来。
            因为在这里，要执行Q = Wq * X，K = Wk * X，V = Wv * X的操作。所以要灌入encoder的输入embedding。其中X的维度是N*d_model
            view()方法主要是为了把维度打平。
        '''
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        ''' 在这里执行 A = (Q * K) / (d_model ** 0.5)
            同时执行 O = A*V；输出的O是N*d_model维度的（这里有可能v和k的维度不一样，还没有想明白先不考虑）
        '''
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        
        # 这里执行残差连接和layerNorm操作（对数据的所有channel执行均值为0方差为1的归一化处理）
        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
