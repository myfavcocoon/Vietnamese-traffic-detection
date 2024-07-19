import torch, cv2
from utils.common import get_model
import torchvision.transforms as transforms
from PIL import Image
from utils.config import Config
import numpy as np



def get_net(cfg,backbone='18'):
    net = get_model(cfg)
    if backbone=='18':
        state_dict = torch.load("C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\model\\culane_res18.pth", map_location='cpu')['model']
    elif backbone=='34':
        state_dict = torch.load("C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\model\\culane_res34.pth", map_location='cpu')['model']
    elif backbone=='ts':
        state_dict = torch.load("C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\model\\tusimple_res34.pth", map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    return net

def config(backbone='18'):
    if backbone=='18':
        args="C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\configs\\culane_res18.py"
    elif backbone=='34':
        args="C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\configs\\culane_res34.py"
    elif backbone=='ts':
        args="C:\\Users\\ASUS\\Desktop\\all\\Ultra-Fast-Lane-Detection-v2\\configs\\tusimple_res34.py"
    cfg = Config.fromfile(args)
    cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
    cfg.col_anchor = np.linspace(0,1, cfg.num_col) 
    return cfg

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590,exist_list=False):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
    if len(coords)>2:
        custom_order = [2, 0, 1, 3]
        new_list = [coords[i] for i in custom_order[:len(coords)]]
    else: new_list=coords
    if exist_list==False:
        
        return new_list
    elif exist_list==True:
        e_list=[]
        for i in range(4):
            if i in row_lane_idx:
                if valid_row[0,:,i].sum() > num_cls_row / 2:
                    e_list.append(1)
                else: e_list.append(0)
            elif i in col_lane_idx:
                if valid_col[0,:,i].sum() > num_cls_col / 4:
                    e_list.append(1)
                else: e_list.append(0)
        return new_list,e_list

def img_preprocess(img,cfg):
    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    #img = Image.open(img_path).convert('RGB')
    img = img_transforms(img)
    img = img[:,-cfg.train_height:,:]
    img = img.unsqueeze(0)
    return img
def viz(img_path,coords,cfg,img_w=1640, img_h=590):
    img_ori=cv2.imread(img_path)
    img_ori=cv2.resize(img_ori,(1640, 590))
    for lane in coords:
        for coord in lane:
            cv2.circle(img_ori,coord,5,(0,255,0),-1)
    img_pil = Image.fromarray(img_ori)
    img_pil.show()

def make_predict(net,img,cfg,out='viz',exist_list=False):
    torch.backends.cudnn.benchmark = True
    cfg.batch_size = 1
    cls_num_per_lane = 18

    pil_img=img_preprocess(img,cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pil_img = pil_img.to(device)
    with torch.no_grad():
        pred = net(pil_img)
    if out=='pred':
        return pred
    elif out=='coords':
        img_w, img_h = 1640, 590
        if exist_list==False:
            coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
            return coords
        elif exist_list==True:
            coords,e_list = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h,exist_list=True)
            return coords,e_list
    elif out=='viz':
        img_w, img_h = 1640, 590
        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
        img_re=cv2.resize(img,(1640, 590))
        for lane in coords:
            for coord in lane:
                cv2.circle(img_re,coord,5,(0,255,0),-1)
        #return img_re