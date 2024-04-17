import torch 
import torch.nn as nn 
from sklearn import metrics 
import time 
import os 
import numpy as np



def adjust_learning_rate(optimizer, epoch, start_lr = 0.0003):
    lr = start_lr * (0.5 **(epoch //50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 
        

def trainer_concate(model, train_dataloader, optimizer, epochs, batch_size, logger, test_dataloader, device, save_path, loss_fn):
    
    model = model.to(device)
    model.train()
    
    best_spike_acc = 0.0
    model_used_time = 0
    start_time = time.time()
    for epoch in range(epochs):

        adjust_learning_rate(optimizer=optimizer, epoch=epoch)
        logger.info("Epoch:{}  Lr:{:.2E}".format(epoch+1,optimizer.state_dict()['param_groups'][0]['lr']))
        for batch_idx, (batch) in enumerate(train_dataloader):
            model_start_time = time.time()
            
            eeg_data = batch['eeg_data'].to(device)
            topoembedding = batch['topoembedding'].to(device)
            label = batch['label_cls'].to(device).to(torch.float32)

            output_cls = model(x= eeg_data.unsqueeze(1), y = topoembedding.unsqueeze(1))
            
            
            loss_all = loss_fn(label, output_cls)
            
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            
            model_used_time += time.time() - model_start_time
            if (batch_idx+1) % 2 == 0: # 每多少个batch打印一次
                end_time = time.time()
                train_speed = (end_time - start_time) / 100
                model_time = model_used_time / (end_time - start_time)
                
                log_str = f'[epoch:{epoch+1} | batch:{batch_idx+1}] | speeds:{train_speed:.3f} sec/batch | gpu_used: {model_time:.3f} | loss_all: {loss_all:.7f}'
                model_used_time = 0
                start_time = end_time

                logger.info(log_str)
                
        if (epoch+1) % 5==0: # 每5个epoch 存储一遍数据
            tmp_save_name = os.path.join(save_path,'ckpt_cls_{}.pth'.format((epoch+1)))
            torch.save(model.state_dict(), tmp_save_name)


            spike_acc, spike_presion, spike_recall, spike_f1, auc_val, loss_val = tester_concate(test_dataloader,model, device, )

            log_str = f'[{epoch+1}/{epochs}] | test_spike_acc: {spike_acc:.4f} | test_spike_presion: {spike_presion:.4f} | test_spike_recall: {spike_recall:.4f} | test_spike_f1score: {spike_f1:4f} | test_auc_val: {auc_val:4f} | loss in validation: {loss_val:.4f} | save to {tmp_save_name}'
            logger.info(log_str)

            if spike_acc > best_spike_acc:
                best_spike_acc = spike_acc
                torch.save(model.state_dict(), os.path.join(save_path,'ckpt_best.pth'))
                logger.info('save {} to best.pth   ...'.format(tmp_save_name))

    return os.path.join(save_path,'ckpt_best.pth')
    
def tester_concate(test_dataloader,model, device, ):
    if test_dataloader is None:
        return 0,0,0,0,0,0
    model = model.to(device)
    model.eval()
    spike_good_count = 0 
    spike_all_count = 0
    spike_tp = 0 
    spike_tp_fp = 0
    spike_tp_fn = 0
    loss_test = 0.
    loss_func = nn.BCELoss()
    
    with torch.no_grad():
        for batch in  test_dataloader:
            eeg_data = batch['eeg_data'].to(device)
            topoembedding = batch['topoembedding'].to(device)
            label = batch['label_cls'].to(device).to(torch.float32).cpu()
            spike_all_count = spike_all_count + label.shape[0]
            
            output_cls = model(x = eeg_data.unsqueeze(1), y = topoembedding.unsqueeze(1)).cpu()    
            loss_test = loss_func(output_cls, label).item() + loss_test
            output_label = (output_cls > 0.5).type(torch.int16).cpu()
                
            spike_good_count += (output_label == label).type(torch.float).sum().cpu().item()
            
            spike_tp_fn += label.type(torch.float).sum().cpu().item() # 所有正例positive  tp+fn
            spike_tp_fp += output_label.type(torch.float).sum().cpu().item() # 所有预测为true tp+fp
            spike_tp += (output_label * label).type(torch.float).sum().cpu().item() # 预测为true，为正例postive
    
    loss_test = loss_test / spike_all_count
    spike_acc = spike_good_count / spike_all_count 
    spike_presion = spike_tp / (spike_tp_fp + np.finfo(np.float32).eps)
    spike_recall = spike_tp / (spike_tp_fn + np.finfo(np.float32).eps)
    spike_f1 = 2 * spike_presion * spike_recall / (spike_presion + spike_recall + np.finfo(np.float32).eps)
    
    fpr,tpr,thresholds = metrics.roc_curve(label,output_label,pos_label=1)
    auc_val = metrics.auc(fpr,tpr)
    # 改回模型状态
    # model = model.to(device)
    model.train()
    
    return spike_acc, spike_presion, spike_recall, spike_f1, auc_val,loss_test