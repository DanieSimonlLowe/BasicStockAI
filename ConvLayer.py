import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, depth: int, layerSize: int):
        super().__init__(self)
        self.convs = [nn.Conv1d(layerSize,layerSize,5) for i in range(depth*2)]

    def foward(self, x):
        for i in range(0,len(self.convs),2):
            x = F.relu(self.convs[i](F.relu(self.convs[i](x)))) + x
            x = F.max_pool1d(x,3)
        return x
        