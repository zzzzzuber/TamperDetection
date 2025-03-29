# 构建样本数据集
# 目标：二分类模型，无需确认篡改区域，只需确认是否存在篡改
# 输入：篡改图像及标准数据集中的图像
# 输出：标签文档

import os

root_path = 'D:/datasets'

def generate_dataset_file():
    categories = os.listdir(path)
    lines = []
    for c in categories:
        imgs_path = root_path + '/CASIA/' + c
        imgs = os.listdir(imgs_path)
        for img in imgs:
            img_path = imgs_path + '/' + img
            if c == 'fake':
                line = img_path + ' ' + '0' + '\n'
            else:
                line = img_path + ' ' + '1' + '\n'
            lines.append(line)  
    with open(root_path+'/CASIA_labels/labels.txt','w')as f:
        f.writelines(lines)

def split_train_test(label_path,path,prob):
    fake_list = []
    real_list = []
    train_list = []
    test_list = []
    with open(label_path,'r')as f:
        lines = f.readlines()
    
    for line in lines:
        if line.split(' ')[-1] == '0': 
            fake_list.append(line)
        else:
            real_list.append(line)

    train_fake_len = int(len(fake_list)*prob)
    train_real_len = int(len(real_list)*prob)
    
    train_list.extend(fake_list[0:train_fake_len])
    train_list.extend(real_list[0:train_real_len])
    test_list.extend(fake_list[train_fake_len+1:])
    test_list.extend(real_list[train_real_len+1:])
    print(len(fake_list),len(real_list))
    print(len(train_list))
    print(len(test_list))
    with open(path+'/train.txt','w')as f:
        f.writelines(train_list)
    with open(path+'/test.txt','w')as fr:
        fr.writelines(test_list)

# generate_dataset_file(root_path)
path = 'D:/datasets/CASIA_labels'
prob = 0.8
label_path = 'D:/datasets/CASIA_labels/labels.txt'
split_train_test(label_path,path,prob)

