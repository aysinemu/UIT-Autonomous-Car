from ultralytics import YOLO
import random
import cv2
import numpy as np
import math
import time
from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket

model = YOLO("/workspace/RoadSeg/weights/best.pt")

cl = 40

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
	vertices = np.array([[(0, 180), (320, 180), (320, 80), (0, 80)]], dtype=np.int32)
    # vertices = np.array([[(230, 80), (320, 80), (320, 0), (230, 0)]], dtype=np.int32)
	masked_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def blue_road(image):
    lower_blue = np.array([cl,0,0]) 
    upper_blue = np.array([cl,0,0]) 
    mask = cv2.inRange(image, lower_blue, upper_blue) 
    res = cv2.bitwise_and(image,image, mask= mask) 
    return res

CHECKPOINT = 130

def segment(link):
    # if you want all classes
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.9

    results = model.predict(link, conf=conf)
    colors = [cl,cl,cl]
    # print(results)
    if results[0].masks is not None:
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                # cv2.polylines(img, points, True, (255, 0, 0), 1)
                color_number = classes_ids.index(int(box.cls[0]))
                cv2.fillPoly(link, points, colors[color_number])
        link = blue_road(link)
        roi = region_selection_road(link)
        image = apply_roi(link, roi)
        link = cv2.cvtColor(link, cv2.COLOR_BGR2GRAY)
        link = (link*(255/np.max(link))).astype(np.uint8)
        h, w = link.shape
        line_row = link[CHECKPOINT, :]
        # cv2.line(link, (0, CHECKPOINT), (w-1, CHECKPOINT), 90, 2)
        flag = True
        min_x = 0
        max_x = 0
        for x, y in enumerate(line_row):
            if y == 255 and flag:
                flag = False
                min_x = x
            elif y == 255:
                max_x = x
        center_row = int((max_x+min_x)/2)
        x1, y1 = center_row, CHECKPOINT
        for pt in link[0, :]: 
            cv2.circle(link, (x1, y1), 5, (179, 179, 179), 3)
        center_of_road = int(math.sqrt(x1*x1+y1*y1))
        # print(center_of_road)
        cv2.imshow("Image", link)
        return center_of_road-45
    else:
        return link.shape[1]//2
        
if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            # segment_image = GetSeg()
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            center_of_road = segment(raw_image)
            center_of_image = raw_image.shape[1] // 2
            deviation = center_of_road - (raw_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.5, i=0.02, d=0.11)
            print("center_of_road = ",center_of_road)
            print("center_of_image = ",center_of_image)
            print("deviation = ",deviation)
            print("angle_setpoint = ",angle_setpoint)
            print("Car x = ",raw_image.shape[1] // 2)
            print("-----------------------------")
            AVControl( speed=30 , angle = angle_setpoint )
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()

