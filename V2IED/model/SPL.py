import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torchinfo import summary


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
        
        x = self._transfer_mean(x.transpose(1,2))
        y = self._transfer_mean(y.transpose(1,2))
        z = self._transfer_mean(z.transpose(1,2))
        
        
        x = (x + y + z)/3
        
        return x

if __name__ == "__main__":
    model = SFL_Net()
    x = torch.randn(1,1,5,5,50)
    device = torch.device("cuda:0")
    model = model.to(device)
    x = x.to(device)
    out = model(x)
    print(model)
    print(out.shape)
    # summary(model, input_size=(1,1,5,5,50))