from ultralytics import YOLO
model = YOLO('/workspace/RoadSeg/weights/best.pt')
from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import math

error_arr = np.zeros(5)
pre_t = time.time()
# MAX_SPEED = 60

def PID(error, p, i, d):
    global pre_t, error_arr
    error_arr[1:] = error_arr[0:-1]
    error_arr[0] = error
    P = error * p
    delta_t = time.time() - pre_t
    pre_t = time.time()
    # Tránh việc chia cho 0 để bớt xuất hiện lỗi NaN
    if delta_t != 0:
        D = (error - error_arr[1]) / delta_t * d
    else:
        D = 0
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    # Đưa về giá trị tuyết đối
    if abs(angle) > 25:
        angle = np.sign(angle) * 25
    return int(angle)

def apply_roi(img, roi):
    # resize ROI to match the original image size
    roi = cv2.resize(src=roi, dsize=(img.shape[1], img.shape[0]))
    
    assert img.shape[:2] == roi.shape[:2]
    
    # scale ROI to [0, 1] => binary mask
    thresh, roi = cv2.threshold(roi, thresh=128, maxval=1, type=cv2.THRESH_BINARY)
    
    # apply ROI on the original image
    new_img = img * roi
    return new_img

def region_selection_road(image):
	mask = np.zeros_like(image) 
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
    # Bottom-left corner: (230, 80) 180 130
    # Bottom-right corner: (320, 80) 320 130
    # Top-right corner: (320, 0)
    # Top-left corner: (230, 0)
	vertices = np.array([[(0, 180), (320, 180), (320, 100), (0, 100)]], dtype=np.int32)
    # vertices = np.array([[(230, 80), (320, 80), (320, 0), (230, 0)]], dtype=np.int32)
	masked_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def Midlane(image):
    roi = region_selection_road(image)
    image = apply_roi(image, roi)
    results = model(image)
    result = results[0]

    for idx, mask in enumerate(result.masks.xy):
        x1, y1, x2, y2 = result.boxes.xyxy[idx].cpu().numpy().astype(int)
	center_of_road = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))/2
	    
    return abs(center_of_road)


if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            # segment_image = GetSeg()

            print(state)
            # resize = cv2.resize(GetRaw(), (800,800))
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)

            # maxspeed = 90, max steering angle = 25
            center_of_road = Midlane(raw_image)
            center_of_image = raw_image.shape[1] // 2
            deviation = center_of_road - (raw_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.5, i=0.02, d=0.11)
            print("center_of_road = ",center_of_road)
            print("center_of_image = ",center_of_image)
            print("deviation = ",deviation)
            print("angle_setpoint = ",angle_setpoint)
            # print("Car x = ",segment_image.shape[1] // 2)
            AVControl( speed=30 , angle = angle_setpoint )
            # 0.3 0.01 0.05 Best condition 30
            #     0.06 0.14
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()

