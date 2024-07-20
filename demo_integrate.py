from utils.func import get_net,config,pred2coords,img_preprocess,viz
from ultralytics import YOLO
import torch
import cv2
import time 

yolo = YOLO("yolov8x.pt")
def make_predict(net,img,cfg = config()):
    torch.backends.cudnn.benchmark = True
    cfg.batch_size = 1
    cls_num_per_lane = 18

    pil_img=img_preprocess(img,cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pil_img = pil_img.to(device)
    with torch.no_grad():
        pred = net(pil_img)
    img_w, img_h = 1640, 590
    coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
    img_re=cv2.resize(img,(1640, 590))
    for lane in coords:
        for coord in lane:
            cv2.circle(img_re,coord,5,(0,255,0),-1)
    return img_re

def new_single_frame_predict(frame):
    height, width, _= frame.shape
    w_factor=1640/width
    h_factor=590/height
    results = yolo.predict (show=False, source=frame,conf=0.25,classes=[2,3,7,9,11])
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()
    output_img = make_predict(net,frame)
    for i in range(len(boxes)):
        x1, y1, x2, y2 =boxes[i]
        x1=x1*w_factor
        x2=x2*w_factor
        y1=y1*h_factor
        y2=y2*h_factor
        x1, y1, x2, y2 = map(int,[x1,y1,x2,y2])
        if (x2-x1)<(1640*0.65):
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{yolo.names[classes[i]]}: {confidences[i]:.2f}'
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output_img
def single_frame_predict(frame):
    re_img = cv2.resize(frame, (1640, 590))
    results = model.predict(show=False, source=re_img,conf=0.25,classes=[2,3,7,9,11])
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    confidences = results[0].boxes.conf.tolist()

    output_img = lane_detector.detect_lanes(re_img)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{model.names[classes[i]]}: {confidences[i]:.2f}'
        cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output_img
def video_predict(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    # FPS variables initialization
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_type = 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #pred = single_frame_predict(frame)
        pred = new_single_frame_predict(frame)
        fps_frame_count += 1
        if fps_frame_count == 1:
            fps_start_time = time.time()
        elif fps_frame_count > 10:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
        
        cv2.putText(pred, f'FPS: {round(fps, 2)}', (10, 30), font, font_scale, font_color, line_type)
        cv2.imshow('Frame', pred)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

net=get_net(config('34'),'34')
path = "C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\testing\\vn1.mp4"
video_predict(path)

