import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchinfo import summary 
from model.MFL import MFL_Net
from model.SPL import SFL_Net


class V2IED(nn.Module):
    def __init__(self, decision_stragety = 'decision_f'):
        super(V2IED,self).__init__()
        """
        @param: decision_stragety: str, default 'decision_f', 'concate', decision_concate_f
        """  
        self.decision_stragety = decision_stragety
           
        self.MFL_Module = MFL_Net()
        self.SFL_Module = SFL_Net()
        if decision_stragety == 'decision_f':
            self.mlp1 = nn.Sequential(
                nn.Linear(100, 1),
                nn.Sigmoid()
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        elif decision_stragety == 'concate':
            self.mlp1 = nn.Sequential(
                nn.Linear(132, 1),
                nn.Sigmoid()
            )
        elif decision_stragety == 'decision_concate_f':
            self.mlp1 = nn.Sequential(
                nn.Linear(100, 1),
                nn.Sigmoid()
            )
            self.mlp2 = nn.Sequential(
            nn.Linear(32, 1),
               nn.Sigmoid()
            )
            self.mlp3 = nn.Sequential(
                nn.Linear(132, 1),
                nn.Sigmoid()
            )
    def _forward_decision_f(self,x,y):

        y = y.squeeze(2)
        x = self.mlp1(x)
        y = self.mlp2(y.squeeze(2))
        return x.squeeze(),y.squeeze(1)
    
    def _forward_concate(self,x,y):

        _out = torch.cat((x,y.transpose(1,2)),dim=2)
        _out = self.mlp1(_out)
        return _out.squeeze()
    
    def _forward_decision_concate_f(self, x, y):
        _out1 = self.mlp1(x)

        _out3 = torch.cat((x,y.transpose(1,2)),dim=2)
        y = y.squeeze(2)
        _out2 = self.mlp2(y)
        _out3 = self.mlp3(_out3)
        return _out1.squeeze(),_out2.squeeze(1),_out3.squeeze()
            
    def forward (self, x, y): # x: time series, y: 3D topo embeddings 
        x = self.MFL_Module(x) # (1,100)
        y = self.SFL_Module(y) # (32, 1)
        y = y.to(x.device)
        if self.decision_stragety == 'decision_f':
            _out1, _out2 = self._forward_decision_f(x,y)
            return _out1,_out2
        elif self.decision_stragety == 'concate':
            _out = self._forward_concate(x,y)
            return _out
        elif self.decision_stragety == 'decision_concate_f':
            _out1,_out2,_out3 = self._forward_decision_concate_f(x,y)
            return _out1,_out2,_out3
        
    
if __name__ == "__main__":
    decision_stragety = 'concate'
    model = V2IED(decision_stragety)
    summary(model, input_size=[(80,19,200),(80,1,5,5,50)]) 