import cv2
from func import *
import os
import matplotlib.pyplot as plt
#cfg=config(backbone='34')
#net = get_net(cfg=cfg,backbone='34')
def tuple_maker(list_of_string):
    output=[]
    for i in range(0,len(list_of_string),2):
        if '-' in list_of_string[i] or '-' in list_of_string[i+1]:
            continue
        pair=(round(float(list_of_string[i])),round(float(list_of_string[i+1])))
        output.append(pair)
    return output
def get_coords(img_dir,net,cfg):
    with open((img_dir[:-4] + '.lines.txt'), 'r', encoding='utf-8') as file:
        lines = [line.rstrip('\n') for line in file.readlines()]
    c_gt=[]
    for line in lines:
        elements = line.split()   
        pairs = tuple_maker(elements)
        c_gt.append(pairs)
    img=cv2.imread(img_dir)
    c_pred,e_pred=make_predict(net,img,cfg,out='coords',exist_list=True)
    return c_gt,c_pred,e_pred
def vis(coords,name,root="C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\testing\\New folder",marker='circle'):
    img=cv2.imread(os.path.join(root,name+'.jpg'))
    img_re=cv2.resize(img,(1640, 590))
    if marker=='circle':
        for lane in coords:
            for coord in lane:
                cv2.circle(img_re,coord,5,(0,255,0),-1)
        plt.imshow(img_re)
    elif marker=='line':
        for lane in coords:
            for i in range(len(lane)-1):
                cv2.line(img_re, lane[i], lane[i+1], (0, 255, 0), thickness=30) 
        plt.imshow(img_re)
#c_gt,c_pred,e_pred=get_coords('00020')
#line="/driver_23_30frame/05171102_0766.MP4/00020.jpg /laneseg_label_w16/driver_23_30frame/05171102_0766.MP4/00020.png 1 1 1 0"
def eval_reader(line,root="D:\\CLRNet\\data\\CULane"):
    splitted=line.split()
    file_name=splitted[0]
    file_dir=root+file_name
    existence_list=[int(x) for x in splitted[-4:]]
    return file_dir,existence_list
def IoU_calc(gt,pred):
    width = 1640
    height = 590
    blank_image_1 = np.zeros((height, width), dtype=np.uint8)
    blank_image_2 = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(gt)-1):
        cv2.line(blank_image_1, gt[i], gt[i+1], 1, thickness=30)
    for i in range(len(pred)-1):
        cv2.line(blank_image_2, pred[i], pred[i+1], 1, thickness=30)
    intersection=np.count_nonzero(np.logical_and(blank_image_1, blank_image_2)==1)
    union=np.count_nonzero(np.logical_or(blank_image_1, blank_image_2)==1)
    IoU=intersection/union
    return IoU
def conmat_calc(e_gt,e_pred,c_gt,c_pred,thresh = 0.3):
    FP=0
    FN=0
    TP=0
    for i in range(4):
        if e_gt[i]==0:
            c_gt.insert(i,False)
        if e_pred[i]==0:
            c_pred.insert(i,False)
    for i in range(4):
        if e_gt[i]==1 and e_pred[i]==1:
            IoU=IoU_calc(c_gt[i],c_pred[i])
            if IoU >= thresh:
                TP+=1
            else: FN+=1
        if e_gt[i]==1 and e_pred[i]==0:
            FN+=1
        if e_gt[i]==0 and e_pred[i]==1:
            FP+=1
    return TP,FP,FN
def F1_score(TP,FP,FN,beta=1):
    precision=(TP/(TP+FP))
    recall=(TP/(TP+FN))
    F1=(1+beta**2)*((precision*recall)/(precision*(beta**2)+recall))
    return F1