import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchinfo import summary

class ResBlock_for_MFL(nn.Module):
    def __init__(self,in_channels,out_channels,k) -> None:
        super(ResBlock_for_MFL,self).__init__()  
        self.batchnorm = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(in_channels =in_channels, out_channels=out_channels,kernel_size=k,padding=k//2)
        self.conv2 = nn.Conv1d(in_channels =in_channels, out_channels=out_channels,kernel_size=k,padding=k//2)
    def forward(self, x):
        y = self.batchnorm(x) 
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.relu(y)
        x = x + y
        return x 
    
    
class MFL_Net(nn.Module):
    def __init__(self,in_channels =19,T_length = 200, out_channels =128, k_in =3,k_out =1,end_channels =1):
        super(MFL_Net,self).__init__()
        self.convfront = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=k_out)
        self.resblocks1 = ResBlock_for_MFL(in_channels=out_channels, out_channels=out_channels,k=k_in)
        self.resblocks2 = ResBlock_for_MFL(in_channels=out_channels, out_channels=out_channels,k=k_in)
        self.resblocks3 = ResBlock_for_MFL(in_channels=out_channels, out_channels=out_channels,k=k_in)
        self.convbehind = nn.Conv1d(in_channels=out_channels, out_channels=end_channels,kernel_size=k_out)
        self.pool1d = nn.AdaptiveAvgPool1d(T_length//2)
        
    def forward(self, TimeS):
        TimeS = self.convfront(TimeS)
        TimeS = self.resblocks1(TimeS)
        TimeS = self.resblocks2(TimeS)
        TimeS = self.resblocks3(TimeS)
        TimeS = self.convbehind(TimeS)
        TimeS = self.pool1d(TimeS)
        return TimeS 

if __name__ == "__main__":
    model = MFL_Net()
    x = torch.randn(1,19,200)
    out = model(x)
    print(out.shape)
    # summary(model, input_size=(1,19,200))
    
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# MFL_Net                                  [1, 19, 100]              --
# ├─Conv1d: 1-1                            [1, 128, 200]             2,560
# ├─ResBlock_for_MFL: 1-2                  [1, 128, 196]             --
# │    └─BatchNorm1d: 2-1                  [1, 128, 200]             256
# │    └─Conv1d: 2-2                       [1, 128, 198]             49,280
# │    └─Conv1d: 2-3                       [1, 128, 196]             49,280
# │    └─LeakyReLU: 2-4                    [1, 128, 196]             --
# ├─ResBlock_for_MFL: 1-3                  [1, 128, 192]             --
# │    └─BatchNorm1d: 2-5                  [1, 128, 196]             256
# │    └─Conv1d: 2-6                       [1, 128, 194]             49,280
# │    └─Conv1d: 2-7                       [1, 128, 192]             49,280
# │    └─LeakyReLU: 2-8                    [1, 128, 192]             --
# ├─ResBlock_for_MFL: 1-4                  [1, 128, 188]             --
# │    └─BatchNorm1d: 2-9                  [1, 128, 192]             256
# │    └─Conv1d: 2-10                      [1, 128, 190]             49,280
# │    └─Conv1d: 2-11                      [1, 128, 188]             49,280
# │    └─LeakyReLU: 2-12                   [1, 128, 188]             --
# ├─Conv1d: 1-5                            [1, 19, 188]              2,451
# ├─AdaptiveAvgPool1d: 1-6                 [1, 19, 100]              --
# ==========================================================================================
# Total params: 301,459
# Trainable params: 301,459
# Non-trainable params: 0
# Total mult-adds (M): 58.04
# ==========================================================================================
# Input size (MB): 0.02
# Forward/backward pass size (MB): 2.02
# Params size (MB): 1.21
# Estimated Total Size (MB): 3.24
# ==========================================================================================    
