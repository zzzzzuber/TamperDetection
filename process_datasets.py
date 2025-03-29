import os
import cv2 

root_path = 'D:/datasets/CASIA/fake'
imgs = os.listdir(root_path)
for img_path in imgs:
    img_path = root_path + '/' + img_path 
    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))