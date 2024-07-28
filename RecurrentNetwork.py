import torch.nn as nn
from ConvLayer import ConvLayer
import torch.nn.functional as F
import torch

class RecurrentNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__(self)
        # purposfully made small

        self.conv_values_input = nn.Conv1d(6,4,3)
        self.convs_values = ConvLayer(6,4)
        # last 2187 days

        self.conv_sheet = nn.Conv1d(11,5)
        self.convs_sheet = ConvLayer(2,5)
        # last 27 quarters

        self.final = nn.Linear(3*4+3*5+16,17)

        self.last = torch.zeros(16)
        
    
    def foward(self, x):
        x0 = F.relu(self.conv_values_input(x[0]))
        x0 = self.convs_values(x0)

        x1 = F.relu(self.conv_sheet(x[1]))
        x1 = self.convs_sheet(x1)

        inp = torch.cat((
            x0,x1,self.last
        ))

        output = self.final(inp)

        self.last = output[:16]

        return output[-1]
        

    