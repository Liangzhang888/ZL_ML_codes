import mne 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from prep import Preprocess
import time
import json
import os
import sys
sys.path.append("/home/zhangliang/zl_code/V2IED/data/dataProcess")
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
from CONSTANTS import DATA_DICT
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def seed_codes(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


context_description = {
    'label_cls':  tf.io.FixedLenFeature([], tf.int64),
    'edf_path': tf.io.FixedLenFeature([], tf.string),
    'time_sec': tf.io.FixedLenFeature([], tf.float32),
    # 'time_length': tf.io.FixedLenFeature([], tf.int64), # 单位 ms 
}
sequence_description = {
    'eeg_data': tf.io.FixedLenSequenceFeature([200], tf.float32),
    # 'topo_matrix': tf.io.FixedLenSequenceFeature([5,5], tf.float32),
    'TfMap': tf.io.FixedLenSequenceFeature([199,200], tf.float32),  
    # 'topo_data': tf.io.FixedLenSequenceFeature([64,64], tf.float32),
}
def _parse_function(exam_proto):
    context, sequence = tf.io.parse_single_sequence_example(exam_proto,context_features=context_description,sequence_features=sequence_description)
    return dict(context, **sequence)

def transform_function(batch_data, device='cpu',  aug = None, topo_sample_rate = 4):
    output_batch = {}
    # batch_data['label_cls'] = np.where(batch_data['label_cls']<0.5, -100, batch_data['label_cls']-1)

    
    
    output_batch['edf_path'] = batch_data['edf_path']
    output_batch['time_sec'] = torch.tensor(batch_data['time_sec']).to(torch.float32)
    output_batch['label_cls'] = torch.tensor(batch_data['label_cls']).to(torch.int64).to(device)
    output_batch['TfMap'] = torch.tensor(batch_data['TfMap']).to(torch.int64).to(device)
    output_batch['eeg_data'] = torch.tensor(batch_data['eeg_data']).to(torch.float32).to(device)
    output_batch['topoembedding'] = TopoEmbedding(np.array(batch_data['eeg_data']), sample_rate = topo_sample_rate)
    
    return output_batch

def TopoEmbedding(eeg_datas, sample_rate = 4):
    TopoEmbeddings = []
    for eeg_data in eeg_datas:
        TopoEmbedding = np.zeros((eeg_data.shape[1]//4, 5, 5 ))
        for i in range(0, eeg_data.shape[1], sample_rate):
            topoinsect  = eeg_data[:,i]
            topomatrix = embed(topoinsect)
            TopoEmbedding[i//sample_rate] = topomatrix
        TopoEmbeddings.append(TopoEmbedding)
    return torch.tensor(TopoEmbeddings).to(torch.float32)
def embed(topoinsect):
    topomatrix = np.zeros((5,5))
    topomatrix[0,0] = (topoinsect[DATA_DICT['Fp1']] + topoinsect[DATA_DICT['F7']]) /2 
    topomatrix[0,1] = topoinsect[DATA_DICT['Fp1']]
    topomatrix[0,2] = (topoinsect[DATA_DICT['Fp1']] + topoinsect[DATA_DICT['Fz']]) /2
    topomatrix[0,3] = topoinsect[DATA_DICT['Fp2']]
    topomatrix[0,4] = (topoinsect[DATA_DICT['Fp2']] + topoinsect[DATA_DICT['F8']]) /2
    topomatrix[1,0] = topoinsect[DATA_DICT['F7']]
    topomatrix[1,1] = topoinsect[DATA_DICT['F3']]
    topomatrix[1,2] = topoinsect[DATA_DICT['Fz']]
    topomatrix[1,3] = topoinsect[DATA_DICT['F4']]
    topomatrix[1,4] = topoinsect[DATA_DICT['F8']]
    topomatrix[2,0] = topoinsect[DATA_DICT['T3']]
    topomatrix[2,1] = topoinsect[DATA_DICT['C3']]
    topomatrix[2,2] = topoinsect[DATA_DICT['Cz']]
    topomatrix[2,3] = topoinsect[DATA_DICT['C4']]
    topomatrix[2,4] = topoinsect[DATA_DICT['T4']]
    topomatrix[3,0] = topoinsect[DATA_DICT['T5']]
    topomatrix[3,1] = topoinsect[DATA_DICT['P3']]
    topomatrix[3,2] = topoinsect[DATA_DICT['Pz']]
    topomatrix[3,3] = topoinsect[DATA_DICT['P4']]
    topomatrix[3,4] = topoinsect[DATA_DICT['T6']]
    topomatrix[4,0] = (topoinsect[DATA_DICT['T5']] + topoinsect[DATA_DICT['O1']]) /2 
    topomatrix[4,1] = topoinsect[DATA_DICT['O1']] 
    topomatrix[4,2] = (topoinsect[DATA_DICT['O1']] + topoinsect[DATA_DICT['Pz']]) /2
    topomatrix[4,3] = topoinsect[DATA_DICT['O2']] 
    topomatrix[4,4] = (topoinsect[DATA_DICT['O2']] + topoinsect[DATA_DICT['T6']]) /2
    
    return topomatrix
    

def get_dataset(files, batch_size=16, repeat=False, cache=False, shuffle=False):
    with tf.device('/cpu:0'):
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=5)
        if cache:
            ds = ds.cache()

        if repeat:
            ds = ds.repeat()

        if shuffle:
            ds = ds.shuffle(1024 * 2)
            opt = tf.data.Options()
            opt.experimental_deterministic = False
            ds = ds.with_options(opt)

        ds = ds.map(_parse_function, num_parallel_calls=5)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1024*2)
    return ds

class TFRecordDataLoader:
    def __init__(self, files, batch_size, total_size, cache=False, repeat=False, shuffle=False, device='cpu',mode='306',aug=''):
        self.ds = get_dataset(
            files, 
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,
            shuffle=shuffle,)
        
        self.total_size = total_size
        self.batch_size = batch_size
        self._iterator = None
        self._count = 0
        self.device = device
        self.aug = aug

        self.transform_func = transform_function

    
    def __iter__(self):
        self._iterator = self.ds.as_numpy_iterator().__iter__()
        self._count = 0
        return self

    def __next__(self):
        if (self.total_size is not None) and (self._count>self.total_size):
            raise StopIteration
        else:
            self._count += self.batch_size
            batch = self._iterator.__next__()
            return self.transform_func(batch, device=self.device, aug=self.aug)

    def __len__(self):
        return self.epoch_size
    
if __name__ =="__main__":
    """
        Test 模块 
    """
    seed_codes()
    files = tf.io.gfile.glob('/data/zhangliang/dyrad/dyrad_processed/*.tfrecord')
    tfrecorder = TFRecordDataLoader(files=files,
                                   batch_size=80,total_size=None,repeat=False)
    for train_batch in tfrecorder:
        batch_TF = train_batch['TfMap']
        batch_label = train_batch['label_cls']
        batch_time_sec = train_batch['time_sec']
        train_eeg_data = train_batch['eeg_data']
        topo_embedding = train_batch['topoembedding']
        
        print(topo_embedding.shape)
        print(train_eeg_data.shape)
        print(batch_TF.shape)
        print(batch_label.shape)
        print(batch_time_sec)