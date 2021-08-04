from collections import defaultdict
from tqdm import tqdm
import re
import jieba
import json
import torch
from sys import platform
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import BertModelTest
from utils import test
from data import DataPrecessForSentence
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def format(text):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~！。？，]+'
    stopwords = ['嗯', '啊', '哦', '的', '了', '吧', '呃', '呢', '噢', '呀', '嘛', '哎', '喂', '唉', '这个', '那个', '就', '那', '就是', '也', '的话', '还']

    line1 = re.sub(r, '', text)
    list1 = list(jieba.cut(line1, cut_all=False))
    list1 = [w for w in list1 if w not in stopwords]
    return list1


def gene_db(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    inv_index = defaultdict(set)
    texts = []
    for i in tqdm(range(len(lines))):
        line = lines[i]
        cuts, text, _ = line.strip().split('\t')
        texts.append(text)
        cuts = cuts.split()
        for c in cuts:
            inv_index[c].add(i)
    return inv_index, texts


def get_candis(text, inv_index, texts):
    text_list = format(text)
    candi = set()
    for c in text_list:
        candi.update(inv_index[c])
    get_candis_sents_id = [i for i in candi]
    get_candis_sents_origin = [texts[i] for i in candi]
    return get_candis_sents_origin, get_candis_sents_id


def test(model, dataloader):
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            if int(labels[0]) == -1:
                labels = None
            _, _, probabilities = model(seqs, masks, segments, labels)
            batch_time += time.time() - batch_start
            batch_prob = probabilities[:,1].cpu().numpy()
            print(batch_prob.shape)
            stop
            all_prob.extend(batch_prob)
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    return batch_time, total_time, all_labels, all_prob


def classify(test_file, text, bert_tokenizer, model, batch_size=32):

    test_data = DataPrecessForSentence(bert_tokenizer, test_file, text)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    batch_time, total_time, all_labels, all_prob = test(model, test_loader)
    # print(all_prob)
    print(len(all_prob))
    return all_prob


def init_model(pretrained_file):
    bert_tokenizer = BertTokenizer.from_pretrained('models_internet/vocabs.txt', do_lower_case=True)
    if platform == "linux" or platform == "linux2":
        checkpoint = torch.load(pretrained_file)
    else:
        checkpoint = torch.load(pretrained_file, map_location=device)
    model = BertModelTest().to(device)
    model.load_state_dict(checkpoint["model"], False)
    return bert_tokenizer, model


def get_sim_sents(text, candis_id, texts, bert_tokenizer, model):
    sim_sents = []
    sim_id = []
    candis_sents = [texts[i] for i in candis_id]
    predicts = classify(candis_sents, text, bert_tokenizer, model)
    stop
    if predict == 1:
        sim_sents.append(sent)
        sim_id.append(i)
    return sim_sents, sim_id


def cluster(text, inv_index, texts):
    candis_sents, candis_id = get_candis(text, inv_index, texts)
    print(len(candis_sents))
    sim_sents, _ = get_sim_sents(text, candis_id, texts)
    print(len(sim_sents))

def cluster_db(inv_index, texts):
    bert_tokenizer, model = init_model("models_internet/best.pth.tar")
    clusters = defaultdict(set)
    used = set()
    for i in range(len(texts)):
        if i in used:
            print(f'已处理，跳过句子{i}')
            continue
        print(f'处理句子{i}， 已处理{len(used)}，未处理{len(texts) - len(used)}')
        cluster = set()
        cluster.add(i)
        text = texts[i]
        _, candis_id = get_candis(text, inv_index, texts)
        print(f'    共{len(candis_id)}条候选')
        candis_id = {i for i in candis_id if i not in used}
        print(f'    去除已处理，剩{len(candis_id)}条候选')
        _, sim_id = get_sim_sents(text, candis_id, texts, bert_tokenizer, model)
        sim_id = set(sim_id)
        print(f'    共{len(candis_id)}条相似语句')
        cluster.update(sim_id)
        used.update(cluster)
        clusters[i] = cluster

    clusters_sents = defaultdict(list)
    for k, v in clusters.items():
        clusters_sents[texts[k]] = [texts[i] for i in v]

    json_str = json.dumps(clusters_sents, indent=4, ensure_ascii=False)
    with open('data/user_texts.cluster', 'w') as f:
        f.write(json_str)



if __name__ == '__main__':

    flag = 1
    inv_index, texts = gene_db('../data/pure_user_texts.txt')

    if flag == 0:
        text = '哦哦你你这个啥什么户的。'
        cluster(text, inv_index, texts)
    else:
        cluster_db(inv_index, texts)


