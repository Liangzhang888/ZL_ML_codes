import random 
import os 
import tensorflow as tf 
import json 
import numpy as np 
import torch 
from data.dataProcess.dataloader import TFRecordDataLoader


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

def cross_validation_dataloader(data_paths, data_usage, folds, cross_seed, batch_size,logger):
    abs_path = os.path.abspath(os.path.abspath(__file__))
    abs_folder = os.path.dirname(abs_path)
    if os.path.exists(os.path.join(abs_folder, 'cross_data.json')):
        # 读取交叉验证数据
        logger.info('cross_data.json exists, load it in {}'.format(os.path.join(abs_folder, 'cross_data.json')))
        with open(os.path.join(abs_folder, 'cross_data.json'), 'r') as f:
            cross_data = json.load(f)
        cross_seeds = [i for i in range(folds)]    
        test_files = cross_data[str(cross_seed)]
        train_files = []
        cross_seeds.remove(cross_seed)
        for train_seed in cross_seeds:
            train_files += cross_data[str(train_seed)]
        
        train_dataloader = TFRecordDataLoader(files=train_files, batch_size=batch_size, total_size=None, repeat=False)
        test_dataloader = TFRecordDataLoader(files=test_files, batch_size=batch_size, total_size=None, repeat=False)
        
        return train_dataloader, test_dataloader
        
    else:
        # 生成交叉验证数据并读取保存
        elem_num = len(data_paths) // folds
        random.shuffle(data_paths)
        cross_data = {}
        for i in range(folds):
            cross_data[i] = data_paths[i*elem_num:(i+1)*elem_num]
        
        with open(os.path.join(abs_folder, 'cross_data.json'), 'w') as f:
            json.dump(cross_data, f)
            
        cross_seeds = [i for i in range(folds)]    
        test_files = cross_data[str(cross_seed)]
        train_files = []
        cross_seeds.remove(cross_seed)    
        
        test_files = cross_data[cross_seed]
        train_files = []
        for train_seed in cross_seeds.remove(cross_seed):
            train_files += cross_data[train_seed]
        
        train_dataloader = TFRecordDataLoader(files=train_files, batch_size=batch_size, total_size=None, repeat=False)
        test_dataloader = TFRecordDataLoader(files=test_files, batch_size=batch_size, total_size=None, repeat=False)
        
        return train_dataloader, test_dataloader
