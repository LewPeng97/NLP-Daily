#!/usr/bin/env python
# coding: utf-8
import os
import glob
from rouge import Rouge



def read_file(filename):
    with open(filename, "r",encoding='utf-8') as f:
        cont = f.read()
    return cont

def gen_sentence(filename):
    cont = read_file(filename)
    cont = "".join(cont.split(" "))
    return cont

def compute_rouge_n(text1, text2, n):
    def ngram(text, n):
        leng = len(text)
        word_dic = {}
        for i in range(0, leng, n):
            start = i
            words = ""
            if leng - start < n:
                break
            else:
                words = text[start: start+n]
                word_dic[words] = 1
        return word_dic
    dic1 = ngram(text1, n)
    dic2 = ngram(text2, n)
    x = 0
    y = len(dic2)
    for w in dic1:
        if w in dic2:
            x += 1
    rouge = x / y
    return rouge if rouge <=1.0 else 1.0


def avg_rouge(ref_dir, dec_dir, n):
    ref_files = os.path.join(ref_dir, "*reference.txt")
    filelist = glob.glob(ref_files)
    scores_list = []
    for ref_file in filelist:
        basename = os.path.basename(ref_file)
        number = basename.split("_")[0]
        dec_file = os.path.join(dec_dir, "{}_decoded.txt".format(number))
        dec_cont = gen_sentence(dec_file)
        ref_cont = gen_sentence(ref_file)


        """第一种Rouge"""
        if n == 'l':
            dec_cont = ''.join([i + ' ' for i in dec_cont])
            ref_cont = ''.join([i + ' ' for i in ref_cont])
            rouge = Rouge()
            score = rouge.get_scores(dec_cont, ref_cont)
            scores_list.append(score[0]['rouge-l']['f'])
        else:
            score = compute_rouge_n(dec_cont, ref_cont, n)
            scores_list.append(score)
        """第二种Rouge"""
        # dec_cont = ''.join([i + ' ' for i in dec_cont])
        # ref_cont = ''.join([i + ' ' for i in ref_cont])
        #
        # rouge = Rouge()
        # score = rouge.get_scores(dec_cont, ref_cont)
        # if n == 1:
        #     scores_list.append(score[0]['rouge-1']['f'])
        # elif n == 2:
        #     scores_list.append(score[0]['rouge-2']['f'])
        # elif n == 'l':
        #     scores_list.append(score[0]['rouge-l']['f'])

    return sum(scores_list) / len(scores_list)


if __name__ == "__main__":
    #root_dir = "./logs/weibo_log/decode_model_495000_20191014_111027/"
    #root_dir = "./logs/weibo/decode_model_510000_20191030_014457/"
    root_dir = "/home/penglu/LewPeng/GDE/finished_files/logs_lstm_transformer_encoder/LCSTS/decode_model_650000_20210105_204105"
    ref_dir = os.path.join(root_dir, "rouge_ref")
    dec_dir = os.path.join(root_dir, "rouge_dec_dir")
    for i in range(1,3):
        print("ROUGE-{} : {:.4}".format(i, avg_rouge(ref_dir, dec_dir, i)))
    str = 'l'


    print("ROUGE-{} : {:.4}".format(str, avg_rouge(ref_dir, dec_dir, str)))