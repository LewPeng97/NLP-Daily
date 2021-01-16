#!/usr/bin/env python
# coding: utf-8
import os
import glob
import re
from rouge import Rouge
from tqdm import tqdm
from main import run

def read_file(filename):
    with open(filename, "r",encoding='utf-8') as f:
        cont = f.read()
    return cont

def gen_sentence(filename):
    cont = read_file(filename)
    # print(cont)
    cont = re.sub(r'<eos>','',cont)
    # cont = re.sub(r'[’!"#$%&\'()*+,-./:：？！《》;<=>?@[\\]^_`{|}~]+', '', cont)
    # print(cont)
    # cont = "".join(cont.split(" "))
    # print(cont)
    return cont

def avg_rouge(ref_dir, dec_dir,n):
    ref_files = os.path.join(ref_dir, "*")
    filelist = glob.glob(ref_files)
    scores_list = []
    for ref_file in filelist:
        basename = os.path.basename(ref_file)
        number = basename.split("_")[0]
        dec_file = os.path.join(dec_dir, "{}".format(number))
        # print('***********************************************')
        # print(number)
        dec_cont = gen_sentence(dec_file)
        ref_cont = gen_sentence(ref_file)
        # print('***********************************************')

        # dec_cont = ''.join([i + ' ' for i in dec_cont])
        # ref_cont = ''.join([i + ' ' for i in ref_cont])

        rouge = Rouge()
        score = rouge.get_scores(dec_cont, ref_cont)
        if n == 1:
            scores_list.append(score[0]['rouge-1']['f'])
        elif n == 2:
            scores_list.append(score[0]['rouge-2']['f'])
        elif n == 'l':
            scores_list.append(score[0]['rouge-l']['f'])

    return sum(scores_list) / len(scores_list)

if __name__ == '__main__':
    model_path = '/home/penglu/LewPeng/TranSummary/lcsts_data/processed_data/model'

    all_model = os.path.join(model_path, "lcsts.s2s.transformer.gpu1.*")
    filelist = glob.glob(all_model)
    dir_list = sorted(filelist, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))

    rouge_save = open('/home/penglu/LewPeng/TranSummary/lcsts_data/processed_data/rouge_save.txt','w',encoding='utf-8')
    # for existing_model_name in tqdm(dir_list):
    existing_model_name = '/home/penglu/LewPeng/TranSummary/lcsts_data/processed_data/model/lcsts.s2s.transformer.final.gpu1.epoch63.2'
    run(existing_model_name)

    root_dir = '/home/penglu/LewPeng/TranSummary/lcsts_data/processed_data/result'
    ref_dir = os.path.join(root_dir,'ground_truth')
    dec_dir = os.path.join(root_dir,'summary')

    rouge_1 = avg_rouge(ref_dir, dec_dir, 1)
    rouge_2 = avg_rouge(ref_dir, dec_dir, 2)
    rouge_l = avg_rouge(ref_dir, dec_dir, 'l')
    print('*****************************************************')
    rouge_save.write("EPOCH : {}, ROUGE-1 : {:.4}, ROUGE-2 : {:.4}, ROUGE-L : {:.4}\n".format(existing_model_name, rouge_1,
                                                                                 rouge_2, rouge_l))
    print("EPOCH : {}, ROUGE-1 : {:.4}, ROUGE-2 : {:.4}, ROUGE-L : {:.4}".format(existing_model_name,rouge_1, rouge_2, rouge_l))
    print('*****************************************************')
    rouge_save.close()