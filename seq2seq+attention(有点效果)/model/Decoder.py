import torch.nn as nn
import configparser



con_fig=configparser.ConfigParser()
con_fig.read('./conf/config.ini')
dec_vocab_size=int(con_fig.get("Encoder", "vocab_size"))
embed_size=int(con_fig.get("Decoder", "embed_size"))
enc_hidden_size=int(con_fig.get("Encoder", "enc_hidden_size"))
dec_hidden_size=int(con_fig.get("Decoder", "dec_hidden_size"))
dropout=float(con_fig.get("Encoder", "drop_out"))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(dec_vocab_size,embed_size)
        # self.embed=nn.Embedding.from_pretrained(matrix_cn)
        # self.embed.weight.requires_grad=False

        self.rnn = nn.GRU(embed_size,dec_hidden_size,batch_first=True) #单层
        self.out = nn.Linear(dec_hidden_size,dec_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self,target,enc_hidden):#target=(batch_size,target_len-1),enc_hidden=(1,batch_size,enc_hidden_size)
        target=self.dropout(self.embed(target))  #target=(batch_size,output_length,embed_size)
        out,hid = self.rnn(target,enc_hidden)#out=(batch_size,target_len,dec_hidden_size),hid=(batch_size,target_len,dec_hidden_size)
        return out,hid

