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

        ''' 在这里，k、q、v的维度是：(batch_size, seq_length, d_model)，三维tensor。注意没有n_head这个维度，参考ScaledDotProductAttention#forward()操作之后的view操作
            d_model是输入的embedding的维度d，seq_length就是是输入序列的长度N，n_head是多头数量
            这里的q、k、v没有n_head这个维度是对的，因为在输入的时候，并不需要知道多头的数量，多头的数量应该是在构建网络层的时候设置的。
        '''
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        ''' 在Layers类里对forward方法使用，是把enc_input作为该方法对q、k、v的赋值传进来。
            因为在这里，要执行Q = Wq * X，K = Wk * X，V = Wv * X的操作。所以要灌入encoder的输入embedding。其中X的维度是N*d_model
            view()方法主要是为了把维度打平。比如1*16的数组，可以通过view(4,4)改变成4*4的矩阵。也可以通过view(4,2,2)变成4*2*2张量。
            view()函数在这里的作用是将张量进行形状变换，将原始张量的形状进行重新排列，以满足后续操作的需求。
            view()函数被用于将张量打平，以便进行后续的线性变换操作。
        '''
        # Pass through the pre-attention projection: b x lq x (n*dv)  这里的(n*dv)也指明了传入的参数只有三维，第三维的大小是n*dv
        # Separate different heads: b x lq x n x dv ,需要通过view变成四维
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # 调整矩阵形状，主要是为了使矩阵乘法成立，变成形状：sz_b, n_head,len_q, d_k

        ''' mask.unsqueeze(1)的作用是在mask张量的第一个维度上增加一个维度。这通常是为了在计算中使用广播操作，以便与其他具有不同维度的张量进行操作。
            在这种情况下，将mask的维度从(sz_b, 1, len_k)扩展为(sz_b, 1, 1, len_k)，以便与q进行注意力计算时进行广播。
        '''
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.为了形状变成(sz_b, 1, 1, len_k)之后可以和attention进行运算

        ''' 在ScaledDotProductAttention#forward()这里执行 A = (Q * K) / (d_model ** 0.5)
            同时执行 O = A*V；输出的O是N*d_model维度的（这里有可能v和k的维度不一样，还没有想明白先不考虑）
            注意：传入的q、k、v经过view和transpose操作之后，已经变成四维tensor了，形状是：sz_b, n_head,len_q, d_k。
        '''
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # 这里的-1，是自适应调整形状，经过transpose和view操作之后，q的形状调整成三维，sz_b * len_q * (n * dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        
        # 这里执行残差连接和layerNorm操作（对数据的所有channel执行均值为0方差为1的归一化处理）
        q += residual

        q = self.layer_norm(q)

        return q, attn

# 这个就是架构图里的FFN模块
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

        ''' 在attention模块里没有非线性操作，但是在FFN中有relu激活函数
            x的形状是：sz_b * len_q * (n * dv)，其中第三维的大小是n*dv,其中n是多头的数量n_head
            这里的操作就是：W2 * Relu(W1 * X)
        '''
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
