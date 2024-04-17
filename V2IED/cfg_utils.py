import os
import sys
import shutil
from time import strftime
import logging
import yaml

from copy import deepcopy
from addict import Dict as adict

FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'

class MyLog(object):
    def __init__(self,filename='baselog'):
        self.logger = logging.getLogger(filename)
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.log_filename = '{0}/{1}.log'.format(filename,strftime("%Y-%m-%d"))

        self.logger.addHandler(self.get_file_handler(self.log_filename))
        self.logger.addHandler(self.get_console_handler())

        self.logger.setLevel(logging.DEBUG)

    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, mode='a+',encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

cfg_yaml = "/home/zhangliang/zl_code/V2IED/cfg.yaml"
with open(cfg_yaml, 'r') as f:
    user_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # _recursive_update(cfg,user_cfg)
    cfg = adict(user_cfg) # 被另一个模块引用

print(cfg)

save_model_path = cfg.train.save_path

# if cfg.mode.name=='train':
#     if os.path.exists(save_model_path):
#         print('The train experiment directory already exists!!!')
#     os.makedirs(save_model_path,exist_ok=True)
#     with open(os.path.join(save_model_path,'cfg.yaml'),'w') as fw:
#         yaml.dump(cfg.to_dict(),fw)
# elif cfg.mode.name=='test':
#     if not os.path.exists(save_model_path):
#         print('The test experiment directory not exists!!!')
#         sys.exit(-1)
# elif cfg.mode.name=='eval':
#     pass
# else:
#     raise ValueError('mode must in ["train","eval","test"]')

if cfg.mode.name == 'train':
    if os.path.exists(save_model_path):
        print('The train experiment directory already exists!!!')
    else:
        print('The train experiment directory not exists!!! Create it now ...')
        os.makedirs(save_model_path,exist_ok=True)
    with open(os.path.join(save_model_path,'cfg.yaml'),'w') as fw:
        yaml.dump(cfg.to_dict(),fw)
elif cfg.mode.name == 'test':
    if not os.path.exists(save_model_path):
        print('The test experiment directory not exists!!!')
        sys.exit(-1)


logger = MyLog(save_model_path).logger # 被另一个模块引用 