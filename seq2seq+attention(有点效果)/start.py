## -*- coding:utf-8 -*-
import torch
from utils import load_data
import numpy as np
from model.Seq2Seq import Seq2Seq,LanguageModelCriterion
from rouge import Rouge
import json
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import time
min_loss=10.0

def evaluate():#验证函数
    global min_loss
    total_num_words = total_loss = 0.
    with torch.no_grad():
        data_dev = load_data.load_string(r'/home/penglu/LewPeng/LCSTS/PART_I训练文件6591.tsv',batch_size=100)
        data_dev = load_data.textrank(data_dev)
        for it,(source,target) in enumerate(data_dev):
            source_token,source_len,target_token,target_len = load_data.sentoken(source,vocab_stoi,target)

            source_token = pad_sequences(source_token,maxlen=max(source_len),padding='post')
            target_token = pad_sequences(target_token,maxlen=max(target_len),padding='post')

            source_token = torch.from_numpy(source_token).to(device).long()
            target_token = torch.from_numpy(target_token).to(device).long()

            target_input = target_token[:,:-1]  # (batch_size,target_len-1)
            target_output = target_token[:,1:]  # (batch_size,target_len-1)

            target_len = torch.from_numpy(np.array(target_len)-1).to(device).long()  # (batch_size)

            pred = model(source_token,target_input)  # (batch_size,target_len,vocab_size)

            mb_out_mask = torch.arange(target_len.max().item(),device=device)[None,:] < target_len[:,None]  # 实际长度的为true,超出实际长度的设为False
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(pred, target_output, mb_out_mask)

            num_words = torch.sum(target_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    avg_loss=total_loss/total_num_words
    if avg_loss<min_loss:
        min_loss=avg_loss
        torch.save(model.state_dict(),'./模型参数保存/textrank训练参数保存文件2epoch')
        print("成功保存模型参数....")

    print("Evaluation loss", avg_loss)

def get_summary(model,source,target,max_length=20):  # source=(1,source_len)
    preds = [1]
    for i in range(max_length):
        a = len(preds)
        pred = model(source,target)#(1,target_len,vocab_size)
        value,idx = pred[0,-1].topk(1,largest=False)  # 这里将当前概率值最大的加入
        preds.append(idx.item())
        if idx.item() == 0:
            break
        target = torch.LongTensor(preds).unsqueeze(0).to(source.device)
    return target

grads={}
def save_grad(name): #查看梯度信息
    global grads
    def save(grad):
        grads[name]=grad
    return save

def train(num_epochs=20):#训练函数

    for epoch in range(num_epochs):
        start =time.time()
        model.train()
        train_data=load_data.load_string(r'/home/penglu/LewPeng/LCSTS/PART_I训练文件2391996.tsv',batch_size=batch_size)
        data=load_data.textrank(train_data,batch_size=batch_size)
        for iter,(source,target) in enumerate(data):


            source_token,source_len,target_token,target_len=load_data.sentoken(source,vocab_stoi,target)#都是列表类型

            source_token=pad_sequences(source_token,maxlen=max(source_len),padding='post')
            target_token=pad_sequences(target_token,maxlen=max(target_len),padding='post')

            source_token = torch.from_numpy(source_token).to(device).long()
            target_token=torch.from_numpy(target_token).to(device).long()

            source_len = torch.from_numpy(np.array(source_len)).to(device).long()#(64)存储的是这一批句子的实际长度

            target_input = target_token[:,:-1]#(batch_size,target_len)
            target_output = target_token[:,1:]#(batch_size,target_len)


            target_len = torch.from_numpy(np.array(target_len)-1).to(device).long()#(batch_size)
            # pred = model(source_token,target_input)  # (batch_size,target_len-1,vocab_size)

            try:
                pred = model(source_token,target_input)  # (batch_size,target_len-1,vocab_size)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda,'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            mb_out_mask = torch.arange(target_len.max().item(),device=device)[None,:] < target_len[:,None] #实际长度的为true,超出实际长度的设为False
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(pred,target_output,mb_out_mask) #开始计算损失函数


            # 更新模型
            # print('更新之前的Decoder的W参数：',model.decoder.rnn.weight_ih_l0)

            optimizer.zero_grad()
            loss.backward()
            # model.decoder.rnn.weight_ih_l0.register_hook(save_grad('decoder_weight_ih_l0')) #查看梯度的变化
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.) #梯度裁剪
            optimizer.step()

            # print("梯度信息：====》：",grads)
            # print('更新之后的Encoder的W参数：',model.decoder.rnn.weight_ih_l0)
            # print('\n\n')
            del source,source_len,target_len,target_input,target_output
            print("Epoch",epoch,"iter",iter,"loss",loss.item(),'time',time.time()-start)
            if iter%1000==0:
                evaluate()

source_to_save = []
summary_to_save = []
model_summary_to_save = []
def predict(source,target):


        source_token,target_token = load_data.sentoken(source,vocab_stoi)

        text_sent = "".join(source[0]) #拿的是验证集的数据进行验证模型效果
        print("文档:",text_sent)
        source_to_save.append(text_sent)

        summary= " ".join(target[0])
        print("摘要：",summary)
        summary_to_save.append(summary)
        source = torch.from_numpy(np.array(source_token).reshape(1, -1)).long().to(device) #(1,source_len)
        # source_len = torch.from_numpy(np.array([len(source_token)])).long().to(device)
        bos = torch.Tensor([[vocab_stoi["<BOS>"]]]).long().to(device) #开头的第一个词

        summary= get_summary(model,source, bos)
        summary = [vocab_itos[i] for i in summary.data.cpu().numpy().reshape(-1)]
        trans = []
        for word in summary[1:]:
            if word != "<EOS>":
                trans.append(word)
            else:
                trans.append("<EOS>")
                break

        pred=" ".join(trans)
        print("模型预测摘要：",pred,"\n")
        model_summary_to_save.append(pred)
        return '<BOS>'+pred+'<EOS>'
if __name__=="__main__":
    batch_size = 256

    f = open(r'./utils/data/vocab_char.json','r')
    vocab_stoi=json.load(f) #加载词典 size:6847
    total_word=len(vocab_stoi)
    print(total_word)

    vocab_itos=dict(zip(vocab_stoi.values(),vocab_stoi.keys()))

    print("词表大小：",total_word)

    device = torch.cuda.set_device(1)
    # matrix_en=preprocessing.embed_matrix('./utils/data/sgns.zhihu.bigram',en_stoi)
    # print(matrix_en)
    #
    # matrix_cn=preprocessing.embed_matrix('./utils/data/sgns.zhihu.bigram',cn_stoi)
    model = Seq2Seq()
    model = model.to(device)
    loss_fn = LanguageModelCriterion().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # train(num_epochs=2)

    model.load_state_dict(torch.load('./模型参数保存/textrank训练参数保存文件2epoch'))


    data_test=load_data.load_string(r"/home/penglu/LewPeng/LCSTS/PART_I训练文件2000.tsv",batch_size=1)
    data_test=load_data.textrank(data_test,batch_size=1)
    rouge=Rouge()
    f1=0.0
    f2=0.0
    f3=0.0
    for iter,(text,summ) in enumerate(data_test):
        pre_target=predict(text,summ)
        target=" ".join(w for w in summ[0])
        rouge_score=rouge.get_scores(target,pre_target)
        f1+=rouge_score[0]['rouge-1']['f']
        f2+=rouge_score[0]['rouge-2']['f']
        f3+=rouge_score[0]['rouge-l']['f']

    print('Rough-1:'+str(f1/(iter+1)))
    print('\n')
    print('Rough-2:'+str(f2/(iter+1)))
    print('\n')
    print('Rough-L:'+str(f3/(iter+1)))
    fopen = open('./1.txt','a')
    fopen.write('2080ti_Rough-1:'+str(f1/(iter+1))+'\n'+'2080ti_Rough-2:'+str(f2/(iter+1))+'\n'+'2080ti_Rough-L:'+str(f3/(iter+1))+'\n')
    file = pd.DataFrame({'文档':source_to_save,'摘要':summary_to_save,'模型预测摘要':model_summary_to_save})
    file.to_csv('./result_all.csv',index=0)