# coding: UTF-8
import time
import torch
import numpy as np
from train_eval3000 import train, init_network,inferone3000
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
args = parser.parse_args()

dataset = 'newsdataset'  # 数据集

model_name = args.model  # bert
x = import_module('models.' + model_name)
config = x.Config(dataset)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load("newsdataset/saved_dict/bert.ckpt"))

def testone3000(text):

    code = build_dataset(config,one=True,text=text)
    code = _to_tensor(config,code)
    # infer
    cls3000 = inferone3000(config, model, code)
    return cls3000
def _to_tensor(config, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(config.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(config.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(config.device)
        return (x, seq_len, mask)