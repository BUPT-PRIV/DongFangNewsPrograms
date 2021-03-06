# coding: UTF-8
import time
import torch
import numpy as np
from train_eval3000 import train, init_network,test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'newsdataset'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    config2 = x.Config(dataset)
    config2.batch_size = 120
    # weight = torch.load(config.save_path)
    # del weight['fc.weight']
    # del weight['fc.bias']
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    # config2.batch_size = 20
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config2)
    test_iter = build_iterator(test_data, config2 )
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load('newsdataset/saved_dict/bert.ckpt'))
    # train(config, model, train_iter, dev_iter, test_iter)
    test(config, model, test_iter)
