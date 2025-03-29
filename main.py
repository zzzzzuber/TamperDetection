import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from model import TamperDetectionCNN
from datasets import TamperDetectionDataset
from utils import AverageMeter
import time
import shutil
import numpy as np
import argparse 
from runtime_parameters import parser

torch.device('cpu')

def main():
    global args,best_prec
    args = parser.parse_args()

    # 基础设置
    BATCH_SIZE = args.batch_size
    epoches = args.epoches
    usePretrainedWeights = args.pretrained_model
    save_path = args.save_path


    # 数据集的label文件
    train_file = 'D:/datasets/CASIA_labels/train.txt'
    test_file = 'D:/datasets/CASIA_labels/test.txt'

    # 数据增强定义
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((128,128)),
        transforms.RandomRotation((0,360)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    ])

    # 数据集定义
    train_dataset = TamperDetectionDataset(train_file,train_transform)
    test_dataset = TamperDetectionDataset(test_file,test_transform)

    # Dataloader定义
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0
    )
    
    #model定义
    model = TamperDetectionCNN()
    if usePretrainedWeights != '':
        checkpoints = torch.load(usePretrainedWeights)
        best_prec = checkpoints['best_pred1']
        model.load_state_dict(checkpoints['state_dict'])
    

    

    # 损失函数定义
    criterion = nn.BCELoss()

    # 定义优化器
    lr = 0.01
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.99,weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,10, gamma=0.1, last_epoch=-1)
    
    # warmup学习率
    
    best_prec = 0
    for epoch in range(epoches):
        
        train(train_loader,model,epoch,criterion,optimizer,lr)
        if epoch % 10 == 0:
            top1 = test(test_loader,model,epoch,criterion)
            is_best = top1 > best_prec
            state = {
                'arch':model,
                'state_dict':model.state_dict(),
                'best_pred1':best_prec
            }
            filename = save_path + 'model_{}.pth'.format(epoch)
            save_checkpoints(state,is_best,filename)
        scheduler.step()


def train(train_loader,model,epoch,criterion,optimizer,lr):
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    batch_time = AverageMeter()
    model.train()
    end = time.time()
    for i,(img,label) in enumerate(train_loader):
        # 获取数据
        label = label.numpy().tolist()
        label = torch.FloatTensor(label)
        img = torch.autograd.Variable(img,requires_grad=False)
        label = torch.autograd.Variable(label,requires_grad=False)
        
        # batch时间
        batch_time.update(time.time()-end)
        
        #模型推理
        pred = model(img)
        pred = torch.argmax(pred,dim=1)
        pred = pred.numpy().tolist()
        pred = torch.FloatTensor(pred)
        
        loss = criterion(pred,label).requires_grad_()

        prec1 = accuracy(pred.data,label)
        

        # 更新参数
        batch_time.update(time.time()-end)
        end = time.time()
        top1.update(prec1.data,pred.size()[0])
        losses.update(loss.data,pred.size()[0])

        # 计算梯度并进行sgd步骤
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        result = "epoch:{epoch},batch_size:{bs}/{len_bs},lr:{lr},loss:{loss:.3f}/{loss_avg:.3f},top1:{top1:.3f}/{top1_avg:.3f},batch_time:{batch_time:.3f}/{batch_time_avg:.3f}".format(
            epoch=epoch,
            bs=i,
            len_bs=len(train_loader),
            lr=lr,
            loss=losses.val,
            loss_avg=losses.avg,
            top1=top1.val,
            top1_avg=top1.avg,
            batch_time=batch_time.val,
            batch_time_avg=batch_time.avg
        )
        print(result)

def test(test_loader,model,epoch,criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i,(img,label)in enumerate(test_loader):
        # 获取数据
        label = label.numpy().tolist()
        label = torch.FloatTensor(label)
        img = torch.autograd.Variable(img,requires_grad=False)
        label = torch.autograd.Variable(label,requires_grad=False)
        # batch时间
        batch_time.update(time.time()-end)
        
        #模型推理
        pred = model(img)
        pred = torch.argmax(pred,dim=1)
        pred = pred.numpy().tolist()
        pred = torch.FloatTensor(pred)

        loss = criterion(pred,label)

        prec1 = accuracy(pred,label)

        # 更新参数
        batch_time.update(time.time()-end)
        end = time.time()
        top1.update(prec1.data,pred.size()[0])
        losses.update(loss.data,pred.size()[0])

        result = "epoch:{epoch},batch_size:{bs}/{len_bs},loss:{loss:.3f}/{loss_avg:.3f},top1:{top1:.3f}/{top1_avg:.3f},batch_time:{batch_time}/{batch_time_avg}".format(
            epoch=epoch,
            bs=i,
            len_bs=len(test_loader),
            loss=losses.val,
            loss_avg=losses.avg,
            top1=top1.val,
            top1_avg=top1.avg,
            batch_time=batch_time.val,
            batch_time_avg=batch_time.avg
        )
        print(result)
        return top1.avg

def save_checkpoints(state,is_best,filename):
    torch.save(state,filename)
    if is_best:
        best_name = filename.reaplce('model.pth','best_prec.pth')
        shutil.copyfile(filename,best_name)
    
def warmup(i,batch,init_learning_rate):
    lr = i*init_learning_rate/batch
    return lr

def adjust_learning_rate(epoch,init_lr):
    lr = init_lr
    if epoch % 10 == 0:
        lr = lr * 0.9
    return lr

def accuracy(output,target,topk=1):
    maxk = max((1,))
    batch_size = target.size(0)
    correct = output.eq(target.data).sum().numpy()
    current_res = correct * 100.0/batch_size
    current_res = torch.tensor(current_res)
    return current_res
    

main()