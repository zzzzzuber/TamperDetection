import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import cv2


class SRM(nn.Module):
    def __init__(self):
        super(SRM,self).__init__()
        filter1 = 1/4*np.array([[0,0,0,0,0],
                                [0,-1,2,-1,0],
                                [0,2,-4,2,0],
                                [0,-1,2,-1,0],
                                [0,0,0,0,0]])

        filter2 = 1/12*np.array([[-1,2,-2,2,-1],
                                [2,-6,8,-6,2],
                                [-2,8,-12,8,-2],
                                [2,-6,8,-6,2],
                                [-1,2,-2,2,-1]])

        filter3 = 1/2 * np.array([[0,0,0,0,0],
                                [0,0,0,0,0],
                                [0,1,-2,1,0],
                                [0,0,0,0,0],
                                [0,0,0,0,0]])
        kernel = torch.zeros([30,3,5,5])
        kernel[:,0,:,:] = torch.FloatTensor(filter1)
        kernel[:,1,:,:] = torch.FloatTensor(filter2)
        kernel[:,2,:,:] = torch.FloatTensor(filter3)

        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        
        
    
    def forward(self,x):
        x = F.conv2d(x,self.weight,padding=0,stride=1)
        return x


if __name__ == '__main__':
    s = SRM()
    img = cv2.imread('D:/datasets/CASIA/fake/Tp_S_NNN_S_N_ind00044_ind00044_01334.tif')  
    # img = cv2.imread(r'D:\datasets\casia2groundtruth-master\CASIA\fake\Au_ani_00001.jpg')
    print(img.shape)     
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.FloatTensor(img).unsqueeze(0)
    img = img.permute(0,3,1,2)
    x = s(img)
    x = x.squeeze()
    x = x.permute(1,2,0).numpy()
    print(x.shape)
    print(x)
    # x = cv2.cvtColor(x,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('D:/datasets/sample_srm.jpg',x)
    # cv2.imshow('test',x)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
   