"""
評估腳本：計算檢測精度和召回率
對應 MATLAB 的 boxScore1.m
"""
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# Import decoder
from utils.yolov2_decode import decode_yolov2

def calculate_box_score(
    detector,
    training_data,
    device,
    conf_threshold=0.3,
    iou_threshold=0.5
):
    """
    計算檢測的精度指標
    
    參數:
        detector: 訓練好的檢測模型
        training_data: Dataset
        device: 計算設備
        conf_threshold: confidence threshold
        iou_threshold: IoU threshold
    """
    detector.eval()
    
    box_n = 0
    candi_n = 0
    desired_box_n = 0
    
    with torch.no_grad():
        for i, data_item in enumerate(training_data):
            # 解析資料項
            if isinstance(data_item, dict):
                image_tensor = data_item.get('image')
                label_boxes = data_item.get('boxes', [])
            else:
                image_tensor, label_boxes = data_item
            
            # 計算期望的框數量
            if isinstance(label_boxes, list):
                desired_box_n += len(label_boxes)
            elif isinstance(label_boxes, np.ndarray):
                desired_box_n += label_boxes.shape[0] if label_boxes.size > 0 else 0
            
            try:
                # 預處理圖像
                img_tensor = preprocess_image(image_tensor).to(device)
                
                # === Grid head decode ===
                # Raw output: (1, A*(5+C), S, S)
                raw = detector(img_tensor.unsqueeze(0))
                
                # Decode
                boxes, scores = decode_yolov2(
                    raw,
                    anchors=np.array([[8.9, 11.2], [14.6, 18.7], [25.5, 28.9], [49.6, 45.7]]),
                    num_classes=1,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    img_size=224
                )

                candi_n += len(boxes)

                label_boxes_arr = (
                    np.array(label_boxes)
                    if isinstance(label_boxes, list)
                    else label_boxes
                )

                matched = np.zeros(len(label_boxes_arr), dtype=bool)

                # 計算 Matches (TP)
                for bbox in boxes:
                    if label_boxes_arr is None or len(label_boxes_arr) == 0:
                        continue

                    ious = compute_iou_xywh(bbox, label_boxes_arr)
                    if ious.size == 0:
                        continue
                        
                    max_iou = np.max(ious)
                    max_idx = np.argmax(ious)

                    if max_iou >= iou_threshold and not matched[max_idx]:
                        box_n += 1
                        matched[max_idx] = True
                
            except Exception as e:
                print(f"處理圖像 {i} 時發生錯誤: {e}")
                continue
    
    return {
        'box_n': box_n,
        'candi_n': candi_n,
        'desired_box_n': desired_box_n,
        'conf_threshold': conf_threshold,
        'iou_threshold': iou_threshold
    }


def compute_iou_xywh(box, boxes):
    """
    計算單一 box 與多個 boxes 的 IoU
    box: (x, y, w, h) normalized
    boxes: (N, 4) normalized
    """
    box = np.asarray(box)[:4]
    boxes = np.asarray(boxes)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    boxes = boxes[:, :4]

    bx, by, bw, bh = box
    box_area = bw * bh

    # Center to Top-Left
    b1x1 = bx - bw / 2
    b1y1 = by - bh / 2
    b1x2 = bx + bw / 2
    b1y2 = by + bh / 2
    
    b2x1 = boxes[:, 0] - boxes[:, 2] / 2
    b2y1 = boxes[:, 1] - boxes[:, 3] / 2
    b2x2 = boxes[:, 0] + boxes[:, 2] / 2
    b2y2 = boxes[:, 1] + boxes[:, 3] / 2

    # Intersection
    x1 = np.maximum(b1x1, b2x1)
    y1 = np.maximum(b1y1, b2y1)
    x2 = np.minimum(b1x2, b2x2)
    y2 = np.minimum(b1y2, b2y2)

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    # Union
    union_area = box_area + (boxes[:, 2] * boxes[:, 3]) - inter_area
    
    return inter_area / (union_area + 1e-6)


def preprocess_image(img, target_size=224):
    """
    預處理圖像用於模型輸入
    """
    from torchvision import transforms
    
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    
    if len(img.shape) == 3 and img.shape[0] in [1, 3, 4] and img.shape[1] > 4:
         img = np.transpose(img, (1, 2, 0))
    
    if img.dtype == np.uint8:
        pass
    elif img.max() > 1.0:
        img = img.astype(np.uint8)
    else:
        img = (img * 255).astype(np.uint8)
    
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(img)
    except Exception as e:
        print(f"pre-process error: {e}")
        return torch.zeros(3, target_size, target_size)
