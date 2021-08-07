# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    crition = nn.BCEWithLogitsLoss(weight=None, size_average=True, reduce=True)
    total_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            one_hot = torch.zeros(outputs.shape[0], outputs.shape[1],device=outputs.device).scatter_(1, labels, 1)
            one_hot.requires_grad_(True)
            loss = crition(outputs, one_hot)
            loss.backward()
            # print(loss)
            optimizer.step()
            
            if total_batch % 60 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                true,ind = torch.sort(true,descending=True)
                true = true.view(-1)
                sorted, indices = torch.sort(outputs.data,descending=True)
                indices = indices[:,:4].cpu().view(-1)
                train_acc = metrics.accuracy_score(true, indices)
                dev_acc, dev_loss = evaluate2(config, model, dev_iter)
                if dev_best_acc <= dev_acc:
                    dev_best_acc = dev_acc
                    print("save_model")
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate2(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    crition = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            one_hot = torch.zeros(outputs.shape[0], outputs.shape[1],device=outputs.device).scatter_(1, labels, 1)
            one_hot.requires_grad_(True)
            loss = crition(outputs, one_hot)
            loss_total += loss
            labels = labels.data.cpu()
            labels,ind = torch.sort(labels,descending=True)
            labels = labels.view(-1).numpy()
            sorted, indices = torch.sort(outputs.data,descending=True)
            indices = indices[:,:4].cpu().view(-1).numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, indices)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def evaluate2(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    jaccard_all = []
    crition = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            one_hot = torch.zeros(outputs.shape[0], outputs.shape[1],device=outputs.device).scatter_(1, labels, 1)
            one_hot.requires_grad_(True)
            loss = crition(outputs, one_hot)
            loss_total += loss
            labels = labels.data.cpu()
            labels,ind = torch.sort(labels,descending=True)
            labels = labels.view(-1).numpy()
            sorted, indices = torch.sort(outputs.data,descending=True)
            indices = indices[:,:4].cpu().view(-1).numpy()
            for i in range(int(indices.shape[0]/4)):
                a = set({labels[i*4],labels[i*4+1],labels[i*4+2],labels[i*4+3]})
                b = set({indices[i*4],indices[i*4+1],indices[i*4+2],indices[i*4+3]})
                bing = len(set(a) | set(b))
                jiao = len(set(a) & set(b))
                jaccard = float(jiao)/float(bing)
                jaccard_all.append(jaccard)
    acc = np.sum(jaccard_all)/len(jaccard_all)
    if test:
        # report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), 'report', 'confusion'
    return acc, loss_total / len(data_iter)

def inferone3000(config, model, text, test=False):
    with open('newsdataset/data/class.txt','r',encoding='utf-8') as f:
        clsnam = f.readlines()
    for i in range(len(clsnam)):
        clsnam[i] = clsnam[i].strip('\n')
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        outputs = model(text)
        sorted, indices = torch.sort(outputs.data,descending=True)
        indices = indices[:,:4].cpu()
    answer = []
    for j in range(4):
        answer.append(clsnam[int(indices[0,j])])
    return answer