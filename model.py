import torch
import torchvision
import torch.nn as nn
import numpy as np 
from srm import SRM



class TamperDetectionCNN(nn.Module):
    def __init__(self):
        super(TamperDetectionCNN,self).__init__()
        self.srm = SRM()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=30,out_channels=16,kernel_size=3,stride=1), # layer 2 [bs,60,60,30]
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2), # layer 3 [bs,30,30,16]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1), # layer 4 [bs,28,28,16]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1), # layer 5 [bs,26,26,16]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=2,stride=1,padding=1), # layer 6 [bs,13,13,16]
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1),  # layer 7 [bs,11,11,16]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1),  # layer 8 [bs,9,9,16]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1),  # layer 9 [bs,7,7,16]
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=1),  # layer 10 [bs,5,5,16]
            nn.ReLU()
        )
        self.classify = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(400,2),
            # nn.ReLU(),
            nn.Softmax()
        )
    
    def forward(self,x):
        # input: [bs,3,128,128]
        x = self.srm(x)
        x = self.features(x)
        bs,c,h,w = x.size()
        x = x.reshape(bs,c*h*w)
        x = self.classify(x)
        return x        
        
if __name__ == '__main__':
    x = torch.randn([16,3,128,128])
    model = TamperDetectionCNN()
    y = model(x)
    print(y)
    print(torch.argmax(y,dim=1).data)