# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2

# class GaussianBlur(nn.Module):
#     def __init__(self):
#         super(GaussianBlur, self).__init__()
#         kernel = [[0.03797616, 0.044863533, 0.03797616],
#                   [0.044863533, 0.053, 0.044863533],
#                   [0.03797616, 0.044863533, 0.03797616]]
#         kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
#         self.weight = nn.Parameter(data=kernel, requires_grad=False)
#         self.weight = kernel
#         print('-------------------',self.weight.size())
 
#     def forward(self, x):

#         print('**********************',x.size())
#         x1 = x[:, 0]
#         print('******************',x1.size())
#         x2 = x[:, 1]
#         x3 = x[:, 2]
#         x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=2)
#         x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=2)
#         x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=2)
#         x = torch.cat([x1, x2, x3], dim=1)
#         return x

# g = GaussianBlur()
# img = cv2.imread(r'D:\datasets\tamper_detection\fake_imgs\img_1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = torch.FloatTensor(img).unsqueeze(0)
# print('==========================',img.size())
# x = g(img)
# print(x)

import torch
x = torch.tensor([10])
y = torch.tensor([5])
z = x.mul(y)
w = (x.mul_(y)).mul_(y)
print(z,x)
print(w,x)