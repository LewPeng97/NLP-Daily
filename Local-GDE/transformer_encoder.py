import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import DEVICE
import config


def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
    wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if list(attn_mask):
            # 给需要mask的地方设置一个负无穷
        	attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=300, num_heads=6, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)#(batch_size,seq_len,self.dim_per_head * num_heads)(32,116,16*8)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)#(batch_size*num_heads,seq_len,self.dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if list(attn_mask):
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, torch.from_numpy(position_encoding).float()))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        input_len = torch.from_numpy(input_len)
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len).numpy().tolist() for len in input_len])
        input_pos = input_pos.to(DEVICE)
        return self.position_encoding(input_pos)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=128, ffn_dim=512, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    """Encoder的一层。"""
    def __init__(self, model_dim=300, num_heads=6, ffn_dim=512, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

class Trm_Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。"""
    def __init__(self,
               vocab_size=config.vocab_size,
               max_seq_len=512,
               num_layers=6,
               model_dim=300,
               num_heads=6,
               ffn_dim=1200,
               prob=0.1):
        super(Trm_Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, prob) for _ in
           range(num_layers)])

        self.embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        init_wt_normal(self.embedding.weight)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.W_O = nn.Linear(model_dim,4*model_dim)
        self.change_c = nn.Linear(model_dim,2*model_dim)
        self.change_c_ = nn.Linear(2*model_dim,model_dim)
        self.W_A = nn.Linear(4*model_dim,4*model_dim)

        self.linear_enc = nn.Sequential(nn.Linear(model_dim, model_dim), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(model_dim, model_dim), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(5 * model_dim, 5 * model_dim), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(5 * model_dim, 4 * model_dim), nn.SELU(), nn.Dropout(p=prob))

    def forward(self, inputs, inputs_len):
        output = self.embedding(inputs) #(batch_size,seq_len,model_dim) (64,seq_len,300)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
        gamm_enc = self.linear_enc(output)
        h = output#(B,Seq_len,dim)
        m = nn.Linear(inputs_len[0],2)
        h = h.transpose(1,2)#(B,dim,Seq_len)
        h = m(h.cpu())#(B,dim,2)
        h = h.permute(2,0,1)


        c = self.change_c_(self.change_c(output))#(B,Seq_len,dim)
        n = nn.Linear(inputs_len[0],2)
        c = c.transpose(1,2)#(B,dim.Seq_len)
        c = n(c.cpu())
        c = c.permute(2,0,1)

        h = h.cuda()
        c = c.cuda()
        hidden = h,c

        output = self.W_O(output)              #(batch_size,seq_len,4*model_dim)model_dim=300

        output = self.linear_out(torch.cat([gamm_enc,output],2))

        output_feature = self.W_A(output)      #(batch_size,seq_len,4*model_dim)model_dim=128
        output_feature = output_feature.view(-1,4*config.d_model)#(batch_size*seq_len,4*model_dim)
        # q = nn.Linear(inputs_len[0],2*config.batch_size )
        # attention = attentions[len(attentions)-1]
        # attention = q(attention.cpu())  # (num_heads*batch_size(),seq_len,2*batch_size())
        # attention = attention.transpose(0,2)#(32,116,256)
        # attention = self.W_A(attention.cuda())
        # attention = attention.contiguous().view(-1,4*config.d_model)

        return output,output_feature,hidden #(B,seq_len,4*model_dim),(B*seq_len,4*model_dim)




