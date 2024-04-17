import mne 
import tensorflow as tf 
import numpy as np 
import torch 
import random
import os 
import json 
import pywt 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from CONSTANTS import CH_NAMES, NEW_CHNAMES, SELECT_CHANNELS
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

seed = 42 
seed_codes(seed)

with open("/data/zhangliang/dyrad/dryad/statistic.json", 'r') as f:
    STATISTICS = json.load(f)
INDEX_NUM = 50
INDEX_LIST =[]
TIMEs = 0.8 
INDEX_DURATION = round(TIMEs / INDEX_NUM,3)
start = 0.000
for i in range(INDEX_NUM):
    index = round(start + i* INDEX_DURATION ,4)
    INDEX_LIST.append(index)

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
def _to_list(value):
    """Check if input value is an array-like object, if not,
       build a list with this value as the only element in it.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return list(value)


def _to_int_list(value):
    """Return a list of integers by dtype conversion."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = [value]
    if not isinstance(value, np.ndarray):
        value = np.array(value)
    value = value.astype(int).tolist()
    return value


def _to_float_list(value):
    """Return a list of floats by dtype conversion."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = [value]
    if not isinstance(value, np.ndarray):
        value = np.array(value)
    value = value.astype(float).tolist()
    return value

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    value = _to_list(value)
    # convert input value to bytes
    for idx in range(len(value)):
        if value[idx] is None:
            value[idx] = ''.encode()
        if not isinstance(value[idx], (str, bytes)):
            # convert to bytes
            value[idx] = str(value[idx]).encode()
        elif isinstance(value[idx], str):
            # encode string to bytes
            value[idx] = value[idx].encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=_to_float_list(value)))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=_to_int_list(value)))


def _encode_float_feature_list(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.

    feature_list = []
    feature_len = value.shape[0] if isinstance(value, np.ndarray) else len(value)
    for idx in range(feature_len):
        v = value[idx]
        try:
            feature = _float_feature(v)
            feature_list.append(feature)
        except Exception as e:
            raise ValueError('Failed to encode a tf.train.Feature, value {}: {}'.format(v, e))
    encoded_feature_list = tf.train.FeatureList()
    encoded_feature_list.feature.extend(feature_list)
    return encoded_feature_list

def _encode_map_feature_list(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    feature_list = []
    feature_len = value.shape[0] if isinstance(value, np.ndarray) else len(value)
    for idx in range(feature_len):
        v = value[idx]
        try:
            feature = _float_feature(v.flatten())
            feature_list.append(feature)
        except Exception as e:
            raise ValueError('Failed to encode a tf.train.Feature, value {}: {}'.format(v, e))
    encoded_feature_list = tf.train.FeatureList()
    encoded_feature_list.feature.extend(feature_list)
    return encoded_feature_list

def _parse_function(exam_proto):
    context, sequence = tf.io.parse_single_sequence_example(exam_proto,context_features=context_description,sequence_features=sequence_description)
    return dict(context, **sequence)

def topomap_fig2feature(input_fig,save_name):
    input_fig.set_size_inches(w=2.24,h=2.24)
    # input_feature = transforms.ToTensor()(input_fig)
    plt.subplots_adjust(left=0.01,
                        bottom=0.01, 
                        right=0.99, 
                        top=0.99, 
                        wspace=0.01, 
                        hspace=0.01)

    canvas = FigureCanvasAgg(input_fig)
    canvas.draw()
    w, h = canvas.get_width_height()

    buf = np.frombuffer(canvas.tostring_argb(),dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    topomap_img = Image.frombytes('RGBA',(w,h),buf.tobytes())
    # topomap_img = topomap_img.convert('L')  #  .crop((0,48,224,272))
    topomap_img = topomap_img.convert('RGB')  #  .crop((0,48,224,272))
    # ? 保存图片
    output_path = os.path.join('/data/zhangliang/eeg_topomap_test/dyrad',save_name)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    topomap_img.save(output_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    topomap_feature = transform(topomap_img)
    plt.close()
    return topomap_feature

class Prep_edf:
    def __init__(self, edf_path, aug):
        self.edf_path = edf_path
        self.aug = aug 
        self.edf_name = os.path.basename(edf_path)
    
    def load_edf(self,): # Obtain specific channels and rename them.
        edf_raw = mne.io.read_raw_edf(self.edf_path,preload=True,verbose=False)
        edf_raw = self.filter_edf(edf_raw)
        edf_raw = edf_raw.pick_channels(CH_NAMES)
        edf_raw.rename_channels(dict(zip(CH_NAMES,SELECT_CHANNELS)))
        
        return edf_raw

    def filter_edf(self, edf_raw, l_freq =1., h_freq = 70., notch = 50., resample = 250.):
        edf_raw = edf_raw.filter(l_freq = l_freq, h_freq = h_freq, verbose =False)
        edf_raw = edf_raw.notch_filter(freqs=notch, verbose=False)
        edf_raw = edf_raw.resample(sfreq = resample,verbose = False)
        return edf_raw
        
    def get_data_dcit(self, edf_raw, time_point, time_length =0.8):
        if self.aug:
            data_dict = self.time_aug(edf_raw, time_point, time_length) # DataAugment strategy is referred to the paper:''V2IED``.
        else:
            data_dict = self.get_data(edf_raw, time_point, time_length)
        return data_dict
    
    def time_aug(self, edf_raw, time_point, time_length, aug_length=0.08): # return a dict {start_sec: float, data: np.array, flipped: int (0 or 1)}
        data_dict = {}
        edf_name = self.edf_name
        sfreq = edf_raw.info['sfreq']
        for i in range(-1,3):
            start_sec = time_point + i*aug_length 
            start_samp = int(start_sec * sfreq)
            end_samp = start_samp + int(time_length * sfreq)
            if start_samp < edf_raw.first_samp:
                print("The start time is out of the range of the edf file {}.".format(edf_name))
                continue
            data_array = edf_raw.get_data(start=start_samp, stop=end_samp) # shape: (n_channels, n_samples)
            # print(data_array.shape)
            # fliped_sec, fliped_data = self.flip_aug(data_array, start_sec) # 是否翻转数据
            data_dict[start_sec] = data_array
            # data_dict[fliped_sec] = fliped_data # 是否存储翻转后的数据

        return data_dict 
    
    def flip_aug(self, data_array, start_sec):
        flipped_data = np.flip(data_array, axis=1)
        filped_sec = 0 - start_sec
        return filped_sec, flipped_data
    
    def get_data(self, edf_raw, time_point, time_length):
        data_dict = {}
        start_samp = int(time_point * edf_raw.info['sfreq'])
        end_samp = start_samp + int(time_length * edf_raw.info['sfreq'])
        data_array = edf_raw.get_data(start=start_samp, stop=end_samp)
        data_array = self._normalize(data_array)
        data_dict[time_point] = data_array
        return data_dict

    def _normalize(self, data_array):
        mean = STATISTICS['mean']
        std = STATISTICS['std']
        data_array = (data_array - mean) / std
        return data_array
    
def CWT_TFMap(data_array):
    sr = 250.
    wavename = 'morl'
    totalscale = 200 
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscale 
    scales = cparam / np.arange(totalscale,1,-1)
    cwtmatrs = []
    
    for data in data_array:
        [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 /sr)
        cwtmatrs.append(abs(cwtmatr))
    # print(np.array(cwtmatrs).shape)
    return cwtmatrs 

def TopologyMap(data_array):
    biosemi_montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=SELECT_CHANNELS, sfreq=250., ch_types='eeg', verbose=False)
    info.set_montage(biosemi_montage)
    eeg_evoke = mne.EvokedArray(data_array, info)
    # print(eeg_evoke.info)
    fig, axes = plt.subplots(nrows=1,ncols=1)
    tmp_topo_feature = []
    tmp_topo_features = torch.tensor([])
    for topo_time in INDEX_LIST:
        fig_eeg = eeg_evoke.plot_topomap(times=[topo_time],\
            axes=axes,ch_type='eeg',res=512,size=1,show=True,colorbar=False,sensors=False,\
                outlines=None,)
        save_name = f'{topo_time}_pos.jpg'
        topomap_feature = topomap_fig2feature(fig_eeg,save_name)
        tmp_topo_feature.append(topomap_feature)
    topomap_features = torch.cat(tmp_topo_feature,dim=0)
    return 0

def _pack_function(data,label,TfMap,edf_path,time_sec):
    context = {}
    context['label_cls'] = _int64_feature(label)
    context['edf_path'] = _bytes_feature(edf_path)
    context['time_sec'] = _float_feature(time_sec)
    
    sequence = {}
    sequence['eeg_data'] = _encode_float_feature_list(data)
    sequence['TfMap'] = _encode_map_feature_list(np.array(TfMap))
    
    
    example = tf.train.SequenceExample()
    
    example.context.CopyFrom(tf.train.Features(feature=context))
    example.feature_lists.CopyFrom(tf.train.FeatureLists(feature_list=sequence))
    return example.SerializeToString()

if __name__ == "__main__":
    raw_data_floder = "/data/zhangliang/dyrad/dryad/"
    result_folder = "/data/zhangliang/dyrad/dyrad_processed_2/"
    edf_files = []
    aug = True
    for file in os.listdir(raw_data_floder):
        if file.endswith('.edf'):
            edf_files.append(os.path.join(raw_data_floder,file))
    with open(os.path.join(raw_data_floder,'file_info.json'), 'r') as fp:
        json_data = json.load(fp)
        
    time_length = 0.8
    """
        开始处理数据
    """
    for edf_file in edf_files:
        edf_name = os.path.basename(edf_file)
        patient_id = edf_name.split('.')[0]
        save_path = f'{result_folder}{patient_id}.tfrecord'
        tf_writer = tf.io.TFRecordWriter(save_path)
        
        annotaed_time, label = json_data[edf_name]
        prep_edf = Prep_edf(edf_file,aug=aug)
        edf_raw = prep_edf.load_edf() 
        time_data_dict = prep_edf.get_data_dcit(edf_raw, annotaed_time, time_length= time_length) # {time_point: data_array} time_point > 0: unfliped data array; time_point < 0: fliped data array 
        for key,value in time_data_dict.items():
            time_point = key 
            data_array = value 
            cwt_matrs = CWT_TFMap(data_array)
            Topo_data = TopologyMap(data_array)
            tf_writer.write(_pack_function(data=data_array,label=label,TfMap=cwt_matrs, \
                            edf_path = edf_file, time_sec = time_point))
            
        tf_writer.close()
        print(f"The file {edf_name} has been processed and saved as {save_path}.")
    print("All files have been processed and saved.")    
        

        
        
    
    
    
    


