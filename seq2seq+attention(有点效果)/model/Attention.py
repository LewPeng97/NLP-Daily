import torch.nn as nn
import configparser
import torch
con_fig=configparser.ConfigParser()
con_fig.read('conf/config.ini')



dec_vocab_size=int(con_fig.get("Encoder", "vocab_size"))
embed_size=int(con_fig.get("Decoder", "embed_size"))
enc_hidden_size=int(con_fig.get("Encoder", "enc_hidden_size"))
dec_hidden_size=int(con_fig.get("Decoder", "dec_hidden_size"))
dropout=float(con_fig.get("Encoder", "drop_out"))


class Attention(nn.Module):
    def __init__(self):
        super(Attention,self).__init__()
        self.linear1=nn.Linear(in_features=2*enc_hidden_size,out_features=dec_hidden_size,bias=False) #W_ih_i
        self.linear2=nn.Linear(in_features=dec_hidden_size,out_features=dec_hidden_size)
        self.tanh=nn.Tanh()
        self.linear3=nn.Linear(in_features=dec_hidden_size,out_features=1)

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.linear_in = nn.Linear(enc_hidden_size * 2,dec_hidden_size,bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size,dec_hidden_size)
    def concat(self,enc_out,dec_out):
        pass

    def dot(self,enc_out,dec_out):
        enc_out=self.linear1(enc_out) #双层换一层
        A=torch.bmm(dec_out,enc_out.transpose(1,2)) #(batch_size,target_len,source_len)
        return torch.softmax(A,dim=2)
    def general(self,enc_out,dec_out):
        pass


    def forward(self,encoder_out,decoder_out,f='dot'):
        #encoder_out=(batch_size,source_len,2*enc_hidden_size)
        #decoder_hid=(batch_size,tatget_len,dec_hidden_size)

        Ht=self.linear1(encoder_out)#(batch_size,source_len,2*enc_hidden_size)==>(batch_size,source_len,dec_hidden_size)

        A=torch.bmm(decoder_out,Ht.transpose(1,2))#(batch_size,target_len,source_len)

        return torch.softmax(A,dim=2)