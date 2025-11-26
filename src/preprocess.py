import cv2
import numpy as np
import torch
from ultralytics import YOLO

def calculate_optical_flow(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 1, 15, 1, 3, 1.2, 0)
    return flow

def process_video_with_yolo(video_path, start_frame, fps, yolo_model_path='yolov8n.pt'):
    box_model = YOLO(yolo_model_path)
    cap = cv2.VideoCapture(video_path)
    
    x_lt, y_lt, w_lt, h_lt = [], [], [], []
    frame_idx = -1
    finish_frame = start_frame + int(fps * 5)
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_idx += 1
        if not ret: break
        if frame_idx <= start_frame or frame_idx > finish_frame: continue
        
        results = box_model(frame, verbose=False, max_det=1)
        for r in results:
            if r.boxes:
                boxes = r.boxes.xywh[0].cpu().numpy()
                x_lt.append(boxes[0]); y_lt.append(boxes[1])
                w_lt.append(boxes[2]); h_lt.append(boxes[3])
    cap.release()
    
    if not x_lt: return None, -1

    x, y, w, h = np.round(np.mean(x_lt)), np.round(np.mean(y_lt)), np.round(np.mean(w_lt)), np.round(np.mean(h_lt))
    
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    sign_history = []
    saved_images = []
    count_int = 0
    sum_vertical_flow = 0
    frame_idx = -1
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_idx += 1
        if not ret: break
        if frame_idx <= start_frame or frame_idx > finish_frame: continue
        
        x_min, y_min = int(x - w / 2), int(y - h / 2)
        x_max, y_max = x_min + int(w), y_min + int(h)
        y_min, x_min = max(0, y_min), max(0, x_min)
        
        frame_cropped = frame[y_min:y_max, x_min:x_max]
        if frame_cropped.size == 0: continue
        frame_resize = cv2.resize(frame_cropped, (224, 224), interpolation=cv2.INTER_AREA)
        
        if prev_frame is not None:
            optical_flow = calculate_optical_flow(prev_frame, frame_resize)
            vertical_mean = np.mean(optical_flow[..., 1])
            sum_vertical_flow += np.abs(vertical_mean)
            current_sign = np.sign(vertical_mean)
            
            if len(sign_history) > 0 and current_sign != sign_history[-1]:
                if len(sign_history) >= 4:
                    saved_images.append(prev_raw_frame)
                    if current_sign < 0: count_int += 1
                sign_history = [current_sign]
            else:
                sign_history.append(current_sign)
        
        prev_frame = frame_resize
        prev_raw_frame = frame_cropped
        
    cap.release()
    
    if len(saved_images) > 0 and sum_vertical_flow >= 40:
        final_image = cv2.resize(np.average(np.array(saved_images), axis=0).astype(np.uint8), (224, 224))
        return final_image, count_int * 12
    
    return None, -1
