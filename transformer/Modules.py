import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

''' Q和K的点乘（dotProduct操作），然后进行norm，除以维度大小的开根号（Scale操作）。结果是没有进行softmax操作的，注意力的分数。
    ScaledDotProductAttention继承torch里的nn.Module模块
'''

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention  定义的起始模块，代码应该从这里开始开启，这里定义了 A = (Q * K.T) / (dim ** 0.5) 和 O = A * V
        使用这个模块来构建层的是：SubLayers.py模块
    '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        ''' 入参q的形状是：sz_b, n_head,len_q, d_k。
            变量k的形状应该是(batch_size, num_heads, seq_length, head_dim)，其中：
            batch_size表示批量大小
            num_heads表示注意力头的数量
            seq_length表示序列长度
            head_dim表示每个注意力头的维度
            transpose(2, 3)操作将k的维度进行转置，将第2维和第3维交换，从而使得在计算注意力时能够正确地进行矩阵乘法操作。
            转置第2,3维度，就是要转置seq_length,head_dim这两个维度，而batch_size，num_heads不变。注意是转置操作，N*d,变成d*N矩阵。

            在公式中，在这里执行的核心操作是：A = (Q * K.T) / (dim ** 0.5)

            将q除以self.temperature是为了缩放注意力权重。通过除以温度参数，
            可以控制softmax函数的输出范围，使得模型更加稳定并且有更好的梯度传播。
            这个步骤有助于确保在计算注意力时不会出现数值上的不稳定性或溢出问题。在这里，temperature被赋值为d_k ** 0.5，注意公式中的d_k ** 0.5和代码中的d_k ** 0.5所在的位置

            attn是形状为(batch_size, num_heads, seq_length, seq_length)的张量。在softmax中的dim=-1表示作用于最后一个维度
            softmax操作是在最后一个维度上进行的，不会改变张量的形状，仅在该维度上进行归一化处理
            v和output的形状都是(batch_size, num_heads, seq_length, head_dim)

            attn.masked_fill(mask == 0, -1e9)，mask的形状和attn一样，在mask==0的位置把attn的值平滑成-1e9
        '''
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))     # A = (Q * K.T) / (dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        ''' 在这里执行公式里的：O = A * V，是在进行softmax归一化之后
            结果就是：对句子上下文的每个维度加权（权重就是attn_score）求和得出的向量
        '''
        output = torch.matmul(attn, v)   # O = A * V

        return output, attn
