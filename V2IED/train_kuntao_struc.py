import numpy as np 
import torch 
import sys 
import os 
import time 
import random
import tensorflow as tf 
from cfg_utils import cfg, logger
from cross_data import cross_validation_dataloader
from TFMap_Model import Multi_concate, SpikeNet
import glob 
import random
from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.train.device.split(':')[-1]
loss_func = torch.nn.BCELoss()


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
    
def optimizer_load(optimizer_name, params):
    
    if optimizer_name == 'Adam':
        return torch.optim.Adam([params])
    elif optimizer_name == 'SGD':
        return torch.optim.SGD([params])
    else:
        raise ValueError('optimizer_name is not in [Adam, SGD, RMSprop]')

def train(model, train_dataloader, test_dataloader, optimizer, save_path, epochs, device):
    model_cls = model.to(device) 
    model_cls.train()
    # all_batch_nums = train_dsloader.__len__() // batch_size # 一个epoch的batch数量
    # log_batch_nums = train_dsloader.__len__() // log_nums # 一个epoch 记录 log_nums次
    
    best_spike_acc = 0.0 
    start_time = time.time() 
    model_used_time = 0 
    for epoch in range(epochs):
        logger.info('epoch:{}'.format(epoch+1))
        for batch_idx, (batch_data) in enumerate(train_dataloader):
            
            model_start_time = time.time()
            # output_cls = model_cls(batch_data['eeg_data'].unsqueeze(1).to(device)) 
            output_cls = model_cls(y = batch_data['TfMap'].to(torch.float32).to(device), x = batch_data['eeg_data'].unsqueeze(1).to(device))
             
                     
            loss_all = loss_func(output_cls, batch_data['label_cls'].to(torch.float32).to(device))

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            model_used_time += time.time()-model_start_time
            
            if (batch_idx+1) % 5 == 0:
                end_time = time.time()
                train_speed = (end_time - start_time) / 5
                model_time = model_used_time / (end_time - start_time)
                
                log_str = f'[epoch:{epoch+1} | batch:{batch_idx+1}] | speeds:{train_speed:.3f} sec/batch | gpu_used: {model_time:.3f} | loss_all: {loss_all:.7f}'
                model_used_time = 0
                start_time = end_time

                logger.info(log_str)

            
        if (epoch+1) % 5==0:
            tmp_save_name = os.path.join(save_path,'ckpt_cls_{}.pth'.format((epoch+1)))
            torch.save(model_cls.state_dict(), tmp_save_name)

            spike_acc, spike_presion, spike_recall, spike_f1, auc_val, loss_val = test(model_cls,test_dataloader, device,)

            log_str = f'[{epoch+1}/{epochs}] | test_spike_acc: {spike_acc:.4f} | test_spike_presion: {spike_presion:.4f} | test_spike_recall: {spike_recall:.4f} | test_spike_f1score: {spike_f1:4f} | test_auc: {auc_val:4f} | loss in validation: {loss_val:.4f} | save to {tmp_save_name}'
            logger.info(log_str)

            if spike_acc > best_spike_acc:
                best_spike_acc = spike_acc
                torch.save(model_cls.state_dict(), os.path.join(save_path,'ckpt_best.pth'))
                logger.info('save {} to best.pth   ...'.format(tmp_save_name))


def test(model, test_dataloader, device):
    if test_dataloader is None:
        return 0, 0, 0
    # model = model.cpu()
    model = model.to(device)
    model.eval()
    spike_good_count = 0 
    spike_all_count = 0
    spike_tp = 0 
    spike_tp_fp = 0
    spike_tp_fn = 0
    loss_val = 0.
    
    with torch.no_grad():
        for batch_data in test_dataloader:

            
            spike_all_count += batch_data['eeg_data'].size(0)

            # output_cls = model(batch_data['eeg_data'].unsqueeze(1).to(device)).cpu()
            output_cls = model(y = batch_data['TfMap'].to(torch.float32).to(device), x = batch_data['eeg_data'].unsqueeze(1).to(device)).cpu()
            
            loss_val = loss_func(output_cls, batch_data['label_cls'].to(torch.float32)).item() + loss_val
            output_label = (output_cls > 0.5).type(torch.int16).cpu()
            

            spike_label = batch_data['label_cls']
            
            spike_good_count += (output_label == spike_label).type(torch.float).sum().cpu().item()
            
            spike_tp_fn += spike_label.type(torch.float).sum().cpu().item() # 所有正例positive  tp+fn
            spike_tp_fp += output_label.type(torch.float).sum().cpu().item() # 所有预测为true tp+fp
            spike_tp += (output_label * spike_label).type(torch.float).sum().cpu().item() # 预测为true，为正例postive
    
    loss_val = loss_val / spike_all_count
    spike_acc = spike_good_count / spike_all_count 
    spike_presion = spike_tp / (spike_tp_fp + np.finfo(np.float32).eps)
    spike_recall = spike_tp / (spike_tp_fn + np.finfo(np.float32).eps)
    spike_f1 = 2 * spike_presion * spike_recall / (spike_presion + spike_recall + np.finfo(np.float32).eps)
    
    fpr,tpr,thresholds = metrics.roc_curve(spike_label,output_label,pos_label=1)
    auc_val = metrics.auc(fpr,tpr)

    model.train()
    
    return spike_acc, spike_presion, spike_recall, spike_f1, auc_val,loss_val
   
if __name__ == "__main__":
    seed_codes()
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
    if decision_stragety == 'concate':
        model = Multi_concate()
        model_test = Multi_concate()
    elif decision_stragety =='single':
        model = SpikeNet()
        model_test = SpikeNet()
    
    prams = {'params':model.parameters()}
    initial_params = cfg.optimizer.params
    params.update(initial_params)
    optimizer_name = cfg.optimizer.name
    optimizer = optimizer_load(optimizer_name, initial_params)
    
    mode = cfg.mode.name
    
    if mode == 'train':
        train(model, train_dataloader, test_dataloader, optimizer, save_path, epochs, device)
        
    test_path = os.path.join(save_path, 'ckpt_best.pth')

    print(test_path)
    model_test.load_state_dict(torch.load(test_path))
    
    test_spike_acc, test_spike_presion, test_spike_recall, spike_f1, auc_val,loss_val = test(dsloader= test_dataloader, model= model_test, device=device)
    logger.info(f'TEST Done! | test_spike_acc: {test_spike_acc:.4f} | test_spike_presion: {test_spike_presion:.4f} | test_spike_recall: {test_spike_recall:.4f} | test_auc: {auc_val:4f} |loss_val: {loss_val:.4f} | test_spike_f1score: {spike_f1:4f}')
