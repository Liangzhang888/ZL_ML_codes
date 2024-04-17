import torch.nn as nn 
import torch 
import torch.nn.functional as F 
import tensorflow as tf 
import numpy as np 
import json 
import yaml 
import glob 
import os 
import random

from cfg_utils import cfg, logger 
from cross_data import cross_validation_dataloader
from V2IED import V2IED
from Multi_Concate import Multi_concate, Satelight, SpikeNet, SFL_Net
from trainer_singlemodal import trainer_singlemodal
from trainer_concate import trainer_concate, tester_concate
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.device.split(':')[-1]


def seed_codes(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
seed_codes()

def optimizer_load(optimizer_name, params):
    
    if optimizer_name == 'Adam':
        return torch.optim.Adam([params])
    elif optimizer_name == 'SGD':
        return torch.optim.SGD([params])
    else:
        raise ValueError('optimizer_name is not in [Adam, SGD, RMSprop]')

if __name__ == "__main__":
    # 读取 训练参数
    save_path = cfg.train.save_path 
    epochs = cfg.train.epochs 
    batch_size = cfg.train.batch_size 
    device = torch.device("cuda:0")
    decision_stragety = cfg.train.decision_stragety
    
    # 读取  数据参数
    data_path = cfg.data.data_path 
    data_usage = cfg.data.data_usage.name
    cross_seed = cfg.data.data_usage.cross_seed
    folds = cfg.data.data_usage.folds
    
    # 读取 优化器参数
    weight_decayway = cfg.optimizer.weight_decayway
    name = cfg.optimizer.name
    params = cfg.optimizer.params
    
    data_paths = glob.glob(data_path)
    train_dataloader, test_dataloader = cross_validation_dataloader(data_paths, data_usage, folds, cross_seed, batch_size, logger)
    if decision_stragety == 'single':
        model = Satelight()
        # model = SpikeNet()
    elif decision_stragety == 'concate':
        model = Multi_concate()
    elif decision_stragety == 'SFL':
        model = SFL_Net()
    
    
    prams = {'params':model.parameters()}
    initial_params = cfg.optimizer.params
    params.update(initial_params)
    optimizer_name = cfg.optimizer.name
    optimizer = optimizer_load(optimizer_name, initial_params)
    
    
    if decision_stragety == 'concate':
        loss_fn = nn.BCELoss()
        best_model_path =  trainer_concate(model, train_dataloader, optimizer, epochs, batch_size, logger, test_dataloader, device, save_path, loss_fn)
    
    if decision_stragety == 'single':
        loss_fn = nn.BCELoss()
        best_model_path =  trainer_singlemodal(model, train_dataloader, optimizer, epochs, batch_size, logger, test_dataloader, device, save_path, loss_fn
                                               , decision_stragety = decision_stragety)
    
    if decision_stragety == 'SFL':
        loss_fn = nn.BCELoss()
        best_model_path =  trainer_singlemodal(model, train_dataloader, optimizer, epochs, batch_size, logger, test_dataloader, device, save_path, loss_fn
                                               ,decision_stragety = decision_stragety)