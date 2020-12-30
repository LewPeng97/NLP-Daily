import torch.nn as nn
import configparser
import torch

enc_fig=configparser.ConfigParser()
enc_fig.read('conf/config.ini')

vocab_size=int(enc_fig.get("Encoder", "vocab_size"))
embed_size=int(enc_fig.get("Encoder", "embed_size"))
enc_hidden_size=int(enc_fig.get("Encoder", "enc_hidden_size"))
dec_hidden_size=int(enc_fig.get("Decoder", "dec_hidden_size"))
dropout=float(enc_fig.get("Encoder", "drop_out"))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size) #不使用预训练的词向量
        # self.embed=nn.Embedding.from_pretrained(matrix_en)
        # self.embed.weight.requires_grad=False

        self.rnn = nn.GRU(embed_size,enc_hidden_size,batch_first=True,bidirectional=True)#双向GRU
        self.dropout = nn.Dropout(dropout)
        self.linear= nn.Linear(enc_hidden_size * 2,dec_hidden_size)

    def forward(self,source): #x=(batch_size,source_len)实际的长度

        embedded = self.dropout(self.embed(source)) #(batch_size,seq_len)======>(batch_size,seq_len,embeddim_size)
        out,hid = self.rnn(embedded) #packed_out=(batch_size,source_len,hidden*2),但是不用hid=(2,batch_size,hidden)


        # _,original_idx = sorted_idx.sort(0,descending=False)  #采用预训练文件
        # out = out[original_idx.long()].contiguous()
        # hid = hid[:,original_idx.long()].contiguous()  # (2,20,300)

        hid = torch.cat([hid[-2],hid[-1]],dim=1)  # (2,batch_size,enc_hidden_size)===>(batch_size,2*enc_hidden_size)
        hid = torch.tanh(self.linear(hid)).unsqueeze(0)  #(batch_size,2*enc_hidden_size)===>(1,batch_size,dec_hidden_size)

        return out,hid #out=(batch_size,seq_len,enc_hidden_size*2),hid=(1,batch_size,dec_hidden_size)