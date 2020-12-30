from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

class PlainEncoder(nn.Module):#编码器
    def __init__(self,vocab_size,hidden_size,dropout=0.2):
        super(PlainEncoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,lengths): #都是tensor类型,x=(batch_size,seq_len(实际上是这批最长的长度)),lengths=(每批样本的实际长度)
        #把batch里面的seq按照长度从大到小进行排序
        sorted_len,sorted_idx = lengths.sort(0,descending=True)
        x_sorted = x[sorted_idx.long()]#句子已经按照长度排序
        embedded = self.dropout(self.embed(x_sorted))

        #把变长的句子高效处理
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded,sorted_len.long().cpu().data.numpy(),
                                                            batch_first=True)#不定长度，保证代码高效
        packed_out,hid = self.rnn(packed_embedded)
        out,_ = nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True)
        _,original_idx = sorted_idx.sort(0,descending=False)
        out = out[original_idx.long()].contiguous()
        hid = hid[:,original_idx.long()].contiguous()

        return out,hid[[-1]]


class PlainDecoder(nn.Module): #解码器
    def __init__(self,vocab_size,hidden_size,dropout=0.2):
        super(PlainDecoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,batch_first=True)
        self.out = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,y,y_lengths,hid):
        sorted_len,sorted_idx = y_lengths.sort(0,descending=True)
        y_sorted = y[sorted_idx.long()]
        hid = hid[:,sorted_idx.long()]

        y_sorted = self.dropout(self.embed(y_sorted))  # batch_size, output_length, embed_size

        packed_seq = nn.utils.rnn.pack_padded_sequence(y_sorted,sorted_len.long().cpu().data.numpy(),batch_first=True)
        out,hid = self.rnn(packed_seq,hid)
        unpacked,_ = nn.utils.rnn.pad_packed_sequence(out,batch_first=True)
        _,original_idx = sorted_idx.sort(0,descending=False)
        output_seq = unpacked[original_idx.long()].contiguous()
        #         print(output_seq.shape)
        hid = hid[:,original_idx.long()].contiguous()

        output = F.log_softmax(self.out(output_seq),-1)

        return output,hid

class PlainSeq2Seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(PlainSeq2Seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,x_lengths,y,y_lengths):
        encoder_out,hid = self.encoder(x,x_lengths)
        output,hid = self.decoder(y=y,y_lengths=y_lengths,hid=hid)
        return output,None

    def translate(self,x,x_lengths,y,max_length=60):
        encoder_out,hid = self.encoder(x,x_lengths)
        preds = []
        batch_size = x.shape[0]
        attns = []
        for i in range(max_length):
            output,hid = self.decoder(y=y,y_lengths=torch.ones(batch_size).long().to(y.device),hid=hid)
            y = output.max(2)[1].view(batch_size,1)
            preds.append(y)

        return torch.cat(preds,1),None


# masked cross entropy loss
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask): #input=(batch_size,seq_len,vocab_size)

        input = input.contiguous().view(-1, input.size(2))# input: (batch_size * seq_len ,vocab_size)
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
            mb_x = torch.from_numpy(mb_x).to(device).long() #一批文档
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long() #每篇文档的长度
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


def train(model,data,num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_num_words = total_loss = 0.
        for it,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(device).long()
            mb_input = torch.from_numpy(mb_y[:,:-1]).to(device).long()
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
    text_sent = "".join([text_itos[w] for w in text_dev_sentoi[i]])
    print("原文档:",text_sent)
    summary_sent = "".join([summary_itos[w] for w in summary_dev_sentoi[i]])
    print("摘要：",summary_sent)

    mb_x = torch.from_numpy(np.array(text_dev_sentoi[i]).reshape(1, -1)).long().to(device)
    mb_x_len = torch.from_numpy(np.array([len(text_dev_sentoi[i])])).long().to(device)

    bos = torch.Tensor([[summary_stoi["<BOS>"]]]).long().to(device)

    translation, attn = model.translate(mb_x, mb_x_len, bos)
    translation = [summary_itos[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation:
        if word != "<EOS>":
            trans.append(word)
        else:
            trans.append("<EOS>")
            break
    print("模型生成的摘要：","".join(trans))



if __name__=="__main__":
    batch_size = 64

    train_text,train_summary = preprocessing.load_data('./utils/data/summ_text.txt') #加载数据
    text_stoi,text_total_word = preprocessing.stoi(train_text)
    summary_stoi,summary_total_ward = preprocessing.stoi(train_summary) #构建词典
    text_itos = dict(zip(text_stoi.values(),text_stoi.keys()))
    summary_itos = dict(zip(summary_stoi.values(),summary_stoi.keys()))

    train_text,dev_text,train_summary,dev_summary=train_test_split(train_text,train_summary,test_size=0.2,random_state=12) #分成训练集与测试集

    text_sentoid,summary_sentoid = preprocessing.sentoi(train_text,train_summary,text_stoi,summary_stoi) #将训练集的句子用id来替换

    train_data = preprocessing.gen_examples(text_sentoid,summary_sentoid,batch_size)



    text_dev_sentoi,summary_dev_sentoi=preprocessing.sentoi(dev_text,dev_summary,text_stoi,summary_stoi) #测试集的句子用id来替换

    dev_data = preprocessing.gen_examples(text_dev_sentoi,summary_dev_sentoi,batch_size) #获取测试集的批次样本

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = 0.2
    hidden_size = 100

    encoder = PlainEncoder(vocab_size=text_total_word,hidden_size=hidden_size,dropout=dropout)
    decoder = PlainDecoder(vocab_size=summary_total_ward,hidden_size=hidden_size,dropout=dropout)

    model = PlainSeq2Seq(encoder,decoder)
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # train(model,train_data,num_epochs=2) #开始模型训练
    # torch.save(model.state_dict(),'seqtoseq2的模型保存文件')

    model.load_state_dict(torch.load('seqtoseq的模型保存文件',map_location='cpu'))
    for i in range(300,350):
        translate_dev(i)
        print()
