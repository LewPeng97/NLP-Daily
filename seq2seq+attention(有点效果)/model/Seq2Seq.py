from .Attention import Attention
from .Encoder import Encoder
from .Decoder import Decoder
import torch.nn as nn
import torch

import configparser

con_fig=configparser.ConfigParser()
con_fig.read('./conf/config.ini')

dec_vocab_size=int(con_fig.get("Encoder", "vocab_size"))
embed_size=int(con_fig.get("Decoder", "embed_size"))
enc_hidden_size=int(con_fig.get("Encoder", "enc_hidden_size"))
dec_hidden_size=int(con_fig.get("Decoder", "dec_hidden_size"))
dropout=float(con_fig.get("Encoder", "drop_out"))


class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq,self).__init__()
        self.attention = Attention()
        self.encoder=Encoder()
        self.decoder=Decoder()
        self.linear1=nn.Linear(in_features=2*enc_hidden_size,out_features=dec_hidden_size,bias=False) #计算h*_t
        self.linear2=nn.Linear(in_features=dec_hidden_size,out_features=dec_hidden_size) #计算h*_t S_t+b

        self.linear3=nn.Linear(in_features=dec_hidden_size,out_features=dec_vocab_size)


    def forward(self,source,target):
        encoder_out,encoder_hid=self.encoder(source)#(batch_size,source_len,enc_hid_size)
        decoder_out,decoder_hid= self.decoder(target,encoder_hid)
        A=self.attention(encoder_out,decoder_out)#(batch_size,target_len,source_len)
        C=torch.bmm(A,encoder_out)  #产生文本背景向量(batch_size,target_len,enc_hid_size)
        x=self.linear1(C)+self.linear2(decoder_out)#(batch_size,target_len,dec_hidden_size)
        P=self.linear3(x)
        return -torch.log_softmax(P,dim=2)#产生P_vocab (batch_size,target_len,dec_vocab_size)




# masked cross entropy loss
class LanguageModelCriterion(nn.Module): #自己的损失函数
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask): ##input=(batch_size,seq_len,vocab_size)、target=(batch_size,target_seq_len)、mask=(batch_size,target_seq_len)

        input = input.contiguous().view(-1, input.size(2))# input=(batch_size*target_seq_len,vocab_size)
        target = target.contiguous().view(-1, 1)#target=(batch_size*seq_len,1)
        mask = mask.contiguous().view(-1, 1)
        output = input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


