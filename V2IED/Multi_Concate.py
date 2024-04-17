import torch.nn as nn 
import torch 
import braindecode 
from torchinfo import summary
import torch.nn.functional as F
from math import sqrt

class SFL_Net(nn.Module):
    def __init__(self,in_channels = 1, out_channels = 32, init_numlayers =50):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,1))
        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,1))
        self.lstm1 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=False) # 单向单隐层， 时序相关性数据维度为50
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm3 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=False)
        self.dowmsample1 = nn.AdaptiveAvgPool1d(init_numlayers//2)
        self.dowmsample2 = nn.AdaptiveAvgPool1d(init_numlayers//4)

        self.mlp = nn.Sequential(
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def _transfer_mean(self, data_tensor):
        tmp_tensor = torch.zeros((data_tensor.shape[0], data_tensor.shape[1], 1))
        for i in range(len(data_tensor)):
            for j in range(len(data_tensor[i])):
                tmp_tensor[i][j] = torch.mean(data_tensor[i][j])
        
        return tmp_tensor

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(1), -1)
        
        y = self.dowmsample1(x)
        z = self.dowmsample2(x)
        x,_ = self.lstm1(x.transpose(1,2))
        y,_ = self.lstm2(y.transpose(1,2))
        z,_ = self.lstm3(z.transpose(1,2))
        
        x = self._transfer_mean(x.transpose(1,2)).to(x.device)
        y = self._transfer_mean(y.transpose(1,2)).to(x.device)
        z = self._transfer_mean(z.transpose(1,2)).to(x.device)
        
        
        x = (x + y + z)/3
        x = x.squeeze(2)
        x = self.mlp(x)
        return x.squeeze(1)

class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v) -> None:
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1.0 / sqrt(dim_k)
    
    def forward(self, x):
        Q = self.q(x)
        K = self.k(x) 
        V = self.v(x) 
        
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1)) ) * self._norm_fact
        output = torch.bmm(atten, V)
        
        return output


class AttentionBlock(nn.Module):
    def __init__(self, n ,N=128, C=32) -> None:
        super().__init__()
        self.n = n
        
        self.MyBlock1 = nn.Sequential(
            Self_Attention(input_dim=C *n, dim_k = N // pow(4,n-1), dim_v=C*n), 
            nn.BatchNorm1d(N // pow(4,n-1)),
            nn.Dropout1d(p=0.2),
        )
        
        self.MyBlock2 = nn.Sequential(
            nn.Linear(C*n*2, C*(n+1)),
            nn.BatchNorm1d(N // pow(4,n-1)),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2),    
        )
        self.MyBlock3 = nn.MaxPool2d(kernel_size=(3*N //pow(4,n)+1 , 3*C*(n+1) +1), padding=(0,0),stride=(1,1))
        self.pad = nn.ZeroPad2d((C*(n+3)//2 , C*(n+3)//2,0,0))
        
    def forward(self, x,):
        _local1_out = self.MyBlock1(x)
        _local2_in = torch.concat((x,_local1_out),dim=2)
        _local2_out = self.MyBlock2(_local2_in)
        _local3_in = torch.concat((_local2_out,self.pad(_local2_in)),dim=2)
        _local3_out = self.MyBlock3(_local3_in)
        
        return _local3_out

class Satelight(nn.Module):
    def __init__(self, c = 16) -> None:
        super(Satelight,self).__init__()
        
        self.Factorized = nn.Sequential(
            nn.Conv2d(1,c,kernel_size=(1,99),stride=1,padding=(0,49)),  # 时域卷积 Factorized
            nn.Conv2d(c,2*c,kernel_size=(19,1),stride=1,padding=(0,0)), # 通道卷积 Factorized
        )
        self.Layer1 = nn.Sequential(
            nn.BatchNorm1d(2*c),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.2),
            nn.MaxPool2d(padding=(0,1),kernel_size=(1,75),stride=(1,1)),
        )
        self.Layer2 = AttentionBlock(n = 1)
        self.Layer3 = AttentionBlock(n = 2)
        self.Layer4 = AttentionBlock(n = 3) 
        self.fea = None
        
        self.classfier = nn.Sequential(
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(256,1),   
        )

        # self.specified_layers_output = []
        # self._register_hook()
        
    def _hook(self, module, input, output):
        self.specified_layers_output.append(output)
    
    def _register_hook(self,):
        self.Layer1.register_forward_hook(hook=self._hook)
        self.Layer2.register_forward_hook(hook=self._hook)
        self.Layer3.register_forward_hook(hook=self._hook)
        self.Layer4.register_forward_hook(hook=self._hook)
    
    def forward(self, x):
        
        x = self.Factorized(x)
        x = x.squeeze(2)
        x = self.Layer1(x)
       
        x = self.Layer2(x.permute(0,2,1))
        # The target net is for (batch,128,32), But raw data's dimension is (batch,32,128).
        x = self.Layer3(x)
        x = self.Layer4(x)
        self.fea = x
        x = self.classfier(x)
        x = x.sigmoid()
        x = x.squeeze(1)
        
        return x 
class ResBlock_Meg1(nn.Module):
  def __init__(self, c=224):
    super(ResBlock_Meg1, self).__init__()

    self.bn_relu = nn.Sequential(
        nn.BatchNorm2d(c),
        nn.ReLU()
    )

    self.block = nn.Sequential(
        nn.Conv2d(c,c,(1,3),padding=(0,1)),
        nn.BatchNorm2d(c),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.Conv2d(c,c,(1,3),stride=(1,2),padding=(0,1))
    )
    self.max_pool = nn.MaxPool2d((1,2))

  def forward(self,x):
    x = self.bn_relu(x)
    residual = self.max_pool(x)
    x = self.block(x)
    return x + residual

class ResBlock_Meg2(nn.Module):
  def __init__(self,nb_ch, ch_up=False, time_down=False, c=64):
    super(ResBlock_Meg2, self).__init__()
    self.time_down = time_down
    self.nb_ch = nb_ch
    self.nb_ch_after = self.nb_ch+c if ch_up else self.nb_ch
    if time_down:
      last_conv =  nn.Conv2d(self.nb_ch,self.nb_ch_after,(1,3),stride=(1,2),padding=(0,1))
    else:
      last_conv =  nn.Conv2d(self.nb_ch,self.nb_ch_after,(1,3),stride=(1,1),padding=(0,1))

    self.block = nn.Sequential(
        nn.BatchNorm2d(self.nb_ch),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.Conv2d(self.nb_ch, self.nb_ch,(1,3),padding=(0,1)),
        nn.BatchNorm2d(self.nb_ch),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        last_conv
    )
    if ch_up:
      if time_down:
        self.residual = nn.Sequential(
          nn.Conv2d(self.nb_ch, self.nb_ch_after,1),
          nn.MaxPool2d((1,2))
        )
      else:
        self.residual = nn.Conv2d(self.nb_ch, self.nb_ch_after,1)
    else:
      if time_down:
        self.residual = nn.MaxPool2d((1,2))
      else:
        self.residual = nn.Identity()

      

  def forward(self,x):
    residual = self.residual(x)
    x = self.block(x)
    return x + residual

# ? resnet18 20 层卷积
# ? spikenet 27 层卷积
# ? resnet34 36 层卷积
# ? resnet50 53 层卷积
class SpikeNet(nn.Module):
  def __init__(self, class_num=1, chnnel_base_size=16, dim_1=19, dim_2=200): # ! 改dim_1 就能切换306和39
    super(SpikeNet, self).__init__()
    self.downsample = nn.AvgPool2d((1,2)) #(1,39,150)
    self.factorized_conv = nn.Sequential(
        nn.Conv2d(1,chnnel_base_size,(dim_1-17,1), padding=(7,0)),
        nn.Conv2d(chnnel_base_size,chnnel_base_size,(1,dim_2//2 -99),padding=(0,14)),
    )
    self.fea = None
    # 输入(batch,16,32,128)
    self.resblock1 = ResBlock_Meg1(chnnel_base_size) # False, True
    
    self.resblock2 = ResBlock_Meg2(chnnel_base_size, False, False, chnnel_base_size)
    self.resblock3 = ResBlock_Meg2(chnnel_base_size, False, True, chnnel_base_size)
    self.resblock4 = ResBlock_Meg2(chnnel_base_size, True, False, chnnel_base_size) # 这个没有将channel扩大一倍
    self.resblock5 = ResBlock_Meg2(chnnel_base_size*2, False, True, chnnel_base_size)

    self.resblock6 = ResBlock_Meg2(chnnel_base_size*2, False, False, chnnel_base_size)
    self.resblock7 = ResBlock_Meg2(chnnel_base_size*2, False, True, chnnel_base_size)
    self.resblock8 = ResBlock_Meg2(chnnel_base_size*2, True, False, chnnel_base_size)
    self.resblock9 = ResBlock_Meg2(chnnel_base_size*3, False, True, chnnel_base_size) 

    self.resblock10 = ResBlock_Meg2(chnnel_base_size*3, False, True, chnnel_base_size)
    self.resblock11 = ResBlock_Meg2(chnnel_base_size*3, True, False, chnnel_base_size)
    
    self.flatten = nn.Flatten()
    self.output = nn.Sequential(
        nn.BatchNorm1d(chnnel_base_size*chnnel_base_size*4*4),
        nn.ReLU(),
        nn.Linear(chnnel_base_size*chnnel_base_size*4*4, 256),
        # nn.Linear(chnnel_base_size*chnnel_base_size*4*4, 256),

        # nn.Linear(chnnel_base_size*chnnel_base_size*4*4, 256),
        # nn.ReLU(),
        # nn.Linear(256, class_num),
    )
    self.classifier= nn.Sequential(
      nn.Linear(256, class_num),
    )
    # self.specified_layers_output = []
    # self._register_hook()
    
  def _hook(self, module, input, output):
      self.specified_layers_output.append(output)
  
  def _register_hook(self):
      self.resblock1.register_forward_hook(hook=self._hook)
      self.resblock4.register_forward_hook(hook=self._hook)
      self.resblock8.register_forward_hook(hook=self._hook)
      self.resblock11.register_forward_hook(hook=self._hook)
  
  
  def forward(self, x):
    x = self.downsample(x)
    x = self.factorized_conv(x)
    x = self.resblock1(x)
    
    x = self.resblock2(x)
    x = self.resblock3(x)
    x = self.resblock4(x)
    
    x = self.resblock5(x)
    x = self.resblock6(x)
    x = self.resblock7(x)
    x = self.resblock8(x)
    
    x = self.resblock9(x)
    x = self.resblock10(x)
    x = self.resblock11(x)
    
    x = self.flatten(x)
    x = self.output(x)

    self.fea = x
    x = self.classifier(x)
    x = x.sigmoid()
    return x.squeeze(1)


class Multi_concate(nn.Module):
    def __init__(self,):
        super(Multi_concate,self).__init__()
        self.modal_1 = Satelight()
        self.modal_2 = SFL_Net()
        
        self.mlp = nn.Sequential(
            nn.Linear(256+32,1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x, y):
        tmp = self.modal_1(x)
        x = self.modal_1.fea 
        y = self.modal_2(y).to(x.device)
        y = y.squeeze(2)
        x = x.flatten(start_dim=1,end_dim=-1)
        _concate = torch.concat((x,y),dim=1)
        
        _out = self.mlp(_concate)
        return _out.squeeze()
if __name__ == "__main__":
    x= torch.randn(80,1,19,200)
    # y = torch.randn(80,1,5,5,50)
    # model = Multi_concate()
    
    model = SpikeNet()
    out=  model(x)
    print(out.shape)
    # decision_stragety = 'concate'
    # model = Multi_concate()
    # summary(model, input_size=[(80,19,200),(80,1,5,5,50)]) 