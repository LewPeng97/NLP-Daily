from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time,math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hidden_size,dec_hidden_size,dropout=0.2):
        super(Encoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size) #不使用预训练的词向量
        # self.embed=nn.Embedding.from_pretrained(matrix_en)
        # self.embed.weight.requires_grad=False

        self.rnn = nn.GRU(embed_size,enc_hidden_size,batch_first=True,bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(enc_hidden_size * 2,dec_hidden_size)

    def forward(self,x,lengths):
        sorted_len,sorted_idx = lengths.sort(0,descending=True)
        x_sorted = x[sorted_idx.long()]
        embedded = self.dropout(self.embed(x_sorted))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)
        packed_out,hid = self.rnn(packed_embedded)
        out,_ = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True)
        _,original_idx = sorted_idx.sort(0,descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:,original_idx.long()].contiguous()

        hid = torch.cat([hid[-2],hid[-1]],dim=1)
        hid = torch.tanh(self.fc(hid)).unsqueeze(0)

        return out,hid


class Attention(nn.Module):
    def __init__(self,enc_hidden_size,dec_hidden_size):
        super(Attention,self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2,dec_hidden_size,bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size,dec_hidden_size)

    def forward(self,output,context,mask):
        # output: batch_size, output_len, dec_hidden_size
        # context: batch_size, context_len, 2*enc_hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size * input_len,-1)).view(batch_size,input_len,
            -1)  # batch_size, context_len, dec_hidden_size

        # context_in.transpose(1,2): batch_size, dec_hidden_size, context_len
        # output: batch_size, output_len, dec_hidden_size
        attn = torch.bmm(output,context_in.transpose(1,2))
        # batch_size, output_len, context_len

        attn.data.masked_fill(mask,-1e6)

        attn = F.softmax(attn,dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn,context)
        # batch_size, output_len, enc_hidden_size

        output = torch.cat((context,output),dim=2)  # batch_size, output_len, hidden_size*2

        output = output.view(batch_size * output_len,-1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size,output_len,-1)
        return output,attn

class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hidden_size,dec_hidden_size,dropout=0.2):
        super(Decoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)

        # self.embed=nn.Embedding.from_pretrained(matrix_cn)
        # self.embed.weight.requires_grad=False
        self.attention = Attention(enc_hidden_size,dec_hidden_size)
        self.rnn = nn.GRU(embed_size,hidden_size,batch_first=True)
        self.out = nn.Linear(dec_hidden_size,vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_mask(self,x_len,y_len): #x_len=(batch_size,) y_len=(batch_size,)
        # a mask of shape x_len * y_len

        device = x_len.device
        max_x_len = x_len.max()
        max_y_len = y_len.max()
        x_mask = torch.arange(max_x_len,device=x_len.device)[None,:] < x_len[:,None]
        y_mask = torch.arange(max_y_len,device=x_len.device)[None,:] < y_len[:,None]

        x_mask=x_mask.long()
        y_mask=y_mask.long()

        mask = (1 - x_mask[:,:,None] * y_mask[:,None,:]).bool()
        return mask

    def forward(self,ctx,ctx_lengths,y,y_lengths,hid):
        sorted_len,sorted_idx = y_lengths.sort(0,descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:,sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted,sorted_len.long().cpu().data.numpy(),
                                                       batch_first=True)
        out,hid = self.rnn(packed_seq,hid)
        unpacked,_ = nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
        _,original_idx = sorted_idx.sort(0,descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        hid = hid[:,original_idx.long()].contiguous()

        mask = self.create_mask(y_lengths,ctx_lengths)

        output,attn = self.attention(output_seq,ctx,mask)
        output = F.log_softmax(self.out(output),-1)

        return output,hid,attn


class Seq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(Seq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,x_lengths,y,y_lengths):
        encoder_out,hid = self.encoder(x,x_lengths)
        output,hid,attn = self.decoder(ctx=encoder_out,ctx_lengths=x_lengths,y=y,y_lengths=y_lengths,hid=hid)
        return output,attn

    def translate(self,x,x_lengths,y,max_length=100):
        encoder_out,hid = self.encoder(x,x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output,hid,attn = self.decoder(ctx=encoder_out,ctx_lengths=x_lengths,y=y,
                                           y_lengths=torch.ones(batch_size).long().to(y.device),hid=hid)
            y = output.max(2)[1].view(batch_size,1)
            preds.append(y)
            attns.append(attn)
        return torch.cat(preds,1),torch.cat(attns,1)


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # input: (batch_size * seq_len) * vocab_size
        input = input.contiguous().view(-1, input.size(2))
        # target: batch_size * 1
        target = target.contiguous().view(-1, 1)
        mask = mask.contiguous().view(-1, 1)
        output = -input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def evaluate(model, data):
    model.eval()
    total_num_words = total_loss = 0.
    with torch.no_grad():
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask  =torch.arange(mb_y_len.max().item(), device=device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)



def train(model,data,num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long() #(64,seq_len)
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()#(64,) 这一批句子的实际长度
            mb_input = torch.from_numpy(mb_y[:,:-1]).to(device).long()#(64,seq_len)
            mb_output = torch.from_numpy(mb_y[:,1:]).to(device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(device).long()
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred,attn = model(mb_x,mb_x_len,mb_input,mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(),device=device)[None,:] < mb_y_len[:,None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred,mb_output,mb_out_mask) #开始计算损失函数

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()

            if it % 100 == 0:
                print("Epoch",epoch,"iteration",it,"loss",loss.item())

        print("Epoch",epoch,"Training loss",total_loss / total_num_words)
        if epoch % 5 == 0:
            evaluate(model,dev_data)
def translate_dev(i):
    en_sent = "".join([en_itos[w] for w in en_dev_sentoi[i]])
    print("文档:",en_sent)
    cn_sent = "".join([cn_itos[w] for w in cn_dev_sentoi[i]])
    print("摘要：","".join(cn_sent))

    mb_x = torch.from_numpy(np.array(en_dev_sentoi[i]).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(en_dev_sentoi[i])])).long().to(device)
    bos = torch.Tensor([[cn_stoi["<BOS>"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    translation = [cn_itos[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "<EOS>":
            trans.append(word)
        else:
            trans.append("<EOS>")
            break
    print("模型预测摘要：","".join(trans))


if __name__=="__main__":
    batch_size = 64

    train_en,train_cn = preprocessing.load_data('./utils/data/summ_text.txt')  # 加载数据
    en_stoi,en_total_word = preprocessing.stoi(train_en)
    cn_stoi,cn_total_word = preprocessing.stoi(train_cn)
    en_itos = dict(zip(en_stoi.values(),en_stoi.keys()))
    cn_itos = dict(zip(cn_stoi.values(),cn_stoi.keys()))

    train_en,dev_en,train_cn,dev_cn = train_test_split(train_en,train_cn,test_size=0.2,random_state=12)  # 分成训练集与测试集

    en_sentoid,cn_sentoid = preprocessing.sentoi(train_en,train_cn,en_stoi,cn_stoi)  # 将句子用id来替换

    # train_data = preprocessing.gen_examples(en_sentoid,cn_sentoid,batch_size)
    # random.shuffle(train_data)

    en_dev_sentoi,cn_dev_sentoi = preprocessing.sentoi(dev_en,dev_cn,en_stoi,cn_stoi)  # 将句子用id来替换
    dev_data = preprocessing.gen_examples(en_dev_sentoi,cn_dev_sentoi,batch_size)  # 获取验证集

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.2
    hidden_size = 100

    # matrix_en=preprocessing.embed_matrix('./utils/data/sgns.zhihu.bigram',en_stoi)
    # print(matrix_en)
    #
    # matrix_cn=preprocessing.embed_matrix('./utils/data/sgns.zhihu.bigram',cn_stoi)


    encoder = Encoder(vocab_size=en_total_word,
                           embed_size=100,
                          enc_hidden_size=hidden_size,
                           dec_hidden_size=hidden_size,
                          dropout=dropout)
    decoder = Decoder(vocab_size=cn_total_word,
                          embed_size=100,
                          enc_hidden_size=hidden_size,
                           dec_hidden_size=hidden_size,
                          dropout=dropout)
    model = Seq2Seq(encoder, decoder)
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # train(model, train_data, num_epochs=400)
    # torch.save(model.state_dict(),'seq2seq+attention的模型保存文件')
    model.load_state_dict(torch.load('seq2seq+attention的模型保存文件',map_location='cpu'))

    for i in range(300,350):
        translate_dev(i)
        print()