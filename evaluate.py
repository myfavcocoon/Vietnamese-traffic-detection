from eva_tool import *
from func import *
import cv2

cfg=config(backbone='34')
net = get_net(cfg=cfg,backbone='34')

eval_list = []
with open("D:\\CLRNet\\data\\CULane\\list\\val_gt.txt", 'r',encoding='utf-8') as file:
    for line in file:
        eval_list.append(line.strip())

TPs=0
FPs=0
FNs=0
for line in eval_list:
    img_dir,e_gt=eval_reader(line)
    img=cv2.imread(img_dir)
    c_gt,c_pred,e_pred=get_coords(img_dir,net,cfg)
    TP,FP,FN= conmat_calc(e_gt,e_pred,c_gt,c_pred)
    TPs+=TP
    FPs+=FP
    FNs+=FN
F1=F1_score(TPs,FPs,FNs,beta=1)
F1