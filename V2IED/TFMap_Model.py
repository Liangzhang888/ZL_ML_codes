import torch 
import torch.nn as nn 
from torchinfo import summary
from torchvision import models


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
# torch.Size([80, 19, 200])
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

# torch.Size([80, 19, 199, 200])
class TFMapNet(nn.Module):
    def __init__(self):
        super(TFMapNet,self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(19,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.model.fc = nn.Linear(512,256)
        
        self.lstm = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 1, batch_first = True, bidirectional = False)
        self.fea = None 
        self.classfier = nn.Sequential(
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.model(x)
        x = x.unsqueeze(1)
        x,_ = self.lstm(x)
        self.fea = x.squeeze(1) 
        x = self.classfier(x.squeeze(1))
        return x.squeeze(1)
     
 
class Multi_concate(nn.Module):
    def __init__(self):
      super().__init__()
      self.modal1 = SpikeNet()
      self.model2 = TFMapNet()
      self.classfier = nn.Sequential(
          nn.Linear(512,1),
        nn.Sigmoid()
      )
    
    def forward(self, x, y):
        _tmp1 = self.modal1(x)
        _tmp2 = self.model2(y)
        x = self.modal1.fea
        y = self.model2.fea
        x = torch.cat([x,y],dim=1)
        _out = self.classfier(x)
        return _out.squeeze(1) 
  
if __name__ == "__main__":
    model = Multi_concate()
    y = torch.randn(80, 19, 199, 200)
    x = torch.rand(80, 1, 19, 200)
    out = model(x, y)
    summary(model, input_size=[(80, 1, 19, 200), (80, 19, 199, 200)])
# ResNet(
#   (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (relu): ReLU(inplace=True)
#   (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#   (layer1): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer2): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer3): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (layer4): Sequential(
#     (0): BasicBlock(
#       (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (downsample): Sequential(
#         (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#         (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       )
#     )
#     (1): BasicBlock(
#       (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#       (relu): ReLU(inplace=True)
#       (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#       (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   (fc): Linear(in_features=512, out_features=1000, bias=True)
# )