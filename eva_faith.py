import argparse
import torch
from load_data import DATA
import os
import re
import numpy as np
import random
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="1"
parser = argparse.ArgumentParser(description='LSTM text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')
parser.add_argument('--minseqlen', type=float, default=3, help='minimum sequence length')
parser.add_argument('-epochs', type=int, default=40, help='number of epochs for training')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-dropout', type=float, default=0.7, help='the probability for dropout')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension')
parser.add_argument('-hidden-dim', type=int, default=300, help='number of hidden dimension')
parser.add_argument('-hidden-layer', type=int, default=1, help='number of hidden layers')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--mode', type=str, default='multichannel', help='available models: rand, static, non-static, multichannel')
parser.add_argument('--gpu', default=-1, type=int, help='0:gpu, -1:cpu')
parser.add_argument('--gpu_id', default='-1', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.manual_seed(args.seed)

if args.gpu > -1:
    args.device = "cuda"
else:
    args.device = "cpu"

def tokenizer(s):
    s_clean = string_clean(s)
    return s_clean.split()

def string_clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"[^A-Za-z0-9\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string

# load data
data = DATA(args, tokenizer)
train_iter = data.train_iter
test_iter = data.test_iter


# vocab
wordvocab = data.TEXT.vocab.itos

# full vocab
word_dic_full = {}
word_invdic_full = {}
for ii, ww in enumerate(wordvocab):
    word_dic_full[ww] = ii
    word_invdic_full[ii] = ww

args.embed_num = len(data.TEXT.vocab)
args.class_num = len(data.LABEL.vocab)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# load model
with open(args.save, 'rb') as f:
    model = torch.load(f, map_location='cpu')


class B:
    text = torch.zeros(1).to(args.device)

def batch_from_str_list(s):
    batch = B()
    nums = np.expand_dims(np.array([word_dic_full[x] for x in s]).transpose(), axis=1)
    batch.text = torch.LongTensor(nums).to(args.device) #cuda()
    return batch

def calc_faithfulness(model, filename, res_file, max_len):
    # read text
    num = 100
    textlist = []
    fealist = []
    delta_dict = {}
    with open(filename,'r') as fp:
        lines = fp.readlines()
    for ii in range(1, len(lines), 2):
        line = lines[ii].split(' ||| ')
        line1 = line[0].split(' >> ')
        line2 = line[1].split(' >> ')
        textlist.append(line1[0].split(' ')[:-1])
        fealist.append(line2[0].split(' ')[:-1])
    # calculate faithfulness
    S_faith, count = 0, 0
    for txt, fea in tqdm(zip(textlist, fealist)):
        txt_ori = txt.copy()
        batch_txt_ori = batch_from_str_list(txt_ori)
        prob_txt_ori = model(batch_txt_ori)

        prob_txt_ori_norm = (np.exp(prob_txt_ori.detach().cpu().numpy()) / np.sum(np.exp(prob_txt_ori.detach().cpu().numpy()), axis=1))
        prob_ori = prob_txt_ori_norm.max()
        pred_ori = prob_txt_ori_norm.argmax()
        
        phrase_idxes = []
        for phrase in fea:
            words = phrase.split('-')
            if len(words)<=max_len:
                phrase_idxes = [int(idx) for idx in words]
                break
        if len(phrase_idxes) == 0:
            continue
        delta = measure_phrase_importance(model, batch_txt_ori, phrase_idxes, pred_ori, prob_ori,num)
        res_file.write(str(delta) + ' ' + str(prob_ori))
        res_file.write('\n')
        count += 1
        # print("Progress:{}%".format(round(count/len(lines)/2)*100), end="\r")
        S_faith += delta
    S_faith /= count
    print(count)
    print('\nFaithfullness score for {} is {:.6f}'.format(filename, S_faith))
    return S_faith

def measure_phrase_importance(model, batch, phrase_idxes, label, prob, num=100):
    input_text = batch.text
    sen_len = len(input_text)

    #generate (word, pos) pairs

    random.seed(0)
    start = phrase_idxes[0]
    end = phrase_idxes[-1]

    delta_sum = 0
    count = 0
    while count<num:
        word_pos_arr = [[i, i] for i in range(sen_len)]
        pos_arr = np.random.choice(np.arange(-1,sen_len+1), end-start+1, True)
        if len(phrase_idxes) == 1:
            if pos_arr[0] == phrase_idxes[0]:
                continue
        if len(phrase_idxes)>1 and np.sum(pos_arr - np.array(phrase_idxes) - (pos_arr[0]-start)) == 0:
            continue
        for i, key in enumerate(range(start,end+1)):
            word_pos_arr[key][1] = pos_arr[i]
        word_pos_arr = sorted(word_pos_arr,key=lambda item:item[1])
        shuffled_texts = [input_text[word] for word, pos in word_pos_arr]
        nums = np.expand_dims(np.array(shuffled_texts).transpose(), axis=1)
        batch.text = torch.LongTensor(nums).to(args.device)  # cuda()
        prob_txt_shuffle = model(batch)
        prob_txt_shuffle_norm = (
                    np.exp(prob_txt_shuffle.detach().cpu().numpy()) / np.sum(np.exp(prob_txt_shuffle.detach().cpu().numpy()), axis=1))
        delta = prob - prob_txt_shuffle_norm[0][label]

        delta_sum += delta
        count += 1
    return delta_sum/num


if __name__ == '__main__':
    max_len = 5
    with open('phrase_inter_lstm_imdb_delta_5.txt','w') as f:
        acd_faith = calc_faithfulness(model, 'phrase_inter_lstm_imdb_idx.txt', f, max_len)


