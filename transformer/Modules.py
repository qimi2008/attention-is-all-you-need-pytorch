import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

''' Q和K的点乘（dotProduct操作），然后进行norm，除以维度大小的开根号（Scale操作）。结果是没有进行softmax操作的，注意力的分数。
    ScaledDotProductAttention继承torch里的nn.Module模块
'''

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        ''' 变量k的形状应该是(batch_size, num_heads, seq_length, head_dim)，其中：
            batch_size表示批量大小
            num_heads表示注意力头的数量
            seq_length表示序列长度
            head_dim表示每个注意力头的维度
            transpose(2, 3)操作将k的维度进行转置，将第2维和第3维交换，从而使得在计算注意力时能够正确地进行矩阵乘法操作。

            将q除以self.temperature是为了缩放注意力权重。通过除以温度参数，
            可以控制softmax函数的输出范围，使得模型更加稳定并且有更好的梯度传播。
            这个步骤有助于确保在计算注意力时不会出现数值上的不稳定性或溢出问题。

            attn是形状为(batch_size, num_heads, seq_length, seq_length)的张量。在softmax中的dim=-1表示作用于最后一个维度
            softmax操作是在最后一个维度上进行的，不会改变张量的形状，仅在该维度上进行归一化处理
            v和output的形状都是(batch_size, num_heads, seq_length, head_dim)
        '''
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
