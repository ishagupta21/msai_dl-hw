import torch

import torch.nn as nn
import torch.nn.functional as F

class MyModelwithBias(nn.Module):
    def __init__(self, layer_size = [512, 512, 512]):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3
        
        
        for s in layer_size:
            layers.append(torch.nn.Linear(c,s, bias = False))
            layers.append(torch.nn.ReLU())
            c = s
        
        # Output layer
        layers.append(torch. nn.Linear(c, 102, bias = False))
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
class MyModelNorm(nn.Module):
    def __init__(self, layer_size = [512, 512, 512]):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3
        
        
        for s in layer_size:
            layers.append(torch.nn.Linear(c,s))
            layers.append(torch.nn.LayerNorm(s))
            layers.append(torch.nn.ReLU())
            c = s
        
        # Output layer
        layers.append(torch. nn.Linear(c, 102))
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Example usage

x = torch.randn(10,3,128,128)


for n in range(10):
    netn = MyModelNorm([512]*n)
    print(f"{netn(x).norm()=}")



##Residual network

class MyModelNorm(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = torch.nn.Linear(in_channels, out_channels)
            self.norm = torch.nn.LayerNorm(out_channels)
            self.relu = torch.nn.ReLU()
            if in_channels != out_channels:
                self.skip = torch.nn.Linear(in_channels, out_channels)
            else:
                self.skip = torch.nn.Identity()        

        def forward(self, x):
            y = self.relu(self.norm(self.linear(x)))
            return self.skip(x) + y
        
    def __init__(self, layer_size = [512, 512, 512]):
        super().__init__()
        layers = []
        layers.append(torch.nn.Flatten())
        c = 128*128*3
        layers.append(torch.nn.Linear(c, 512, bias=False))
        c = 512
        for s in layer_size:
            layers.append(self.Block(c,s))
            c = s
        
        # Output layer
        layers.append(torch. nn.Linear(c, 102))
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    




# Example usage

x = torch.randn(10,3,128,128)


for n in range(10):
    netn = MyModelNorm([512]*n)
    print(f"{netn(x).norm()=}")


