import argparse

parser = argparse.ArgumentParser(description="SRM plugin AlexNet for tamper detection")
parser.add_argument('pretrained_model',type=str,default='D:/myProjects/work/tamper_detection/models/best_prec.pth')
parser.add_argument('epoches',type=int,default=100)
parser.add_argument('batch_size',type=int,default=100)
parser.add_argument('save_path',type=str,default='D:/myProjects/work/tamper_detection/models/')