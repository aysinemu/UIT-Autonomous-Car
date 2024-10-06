from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

model = YOLO("/workspace/SignDetect/weights/best.pt")

center_sign_x = 0
center_sign_y = 0 
car_y = 160
car_x = 160
current_speed = 0

error_arr = np.zeros(5)
pre_t = time.time()
# MAX_SPEED = 60

class PIDD:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp  
        self.Ki = Ki  
        self.Kd = Kd  
        self.setpoint = setpoint  
        self.prev_error = 0  
        self.integral = 0 
        self.max_output = 10
    def compute(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error
        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        output = np.clip(output, 0, self.max_output)
        return output

def set_motor_speed(speed):
    global current_speed
    current_speed = speed

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

CHECKPOINT = 160

def Midlane(image):
    roi = region_selection_road(image)
    image = apply_roi(image, roi)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray*(255/np.max(gray))).astype(np.uint8)
    # cv2.imshow("test", gray)
    h, w = gray.shape
    line_row = gray[CHECKPOINT, :]
    # print(line_row)
    # gray = cv2.line(gray, (0, CHECKPOINT), (w-1, CHECKPOINT), 90, 2)
    # cv2.imshow('test', gray)
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
    # gray = cv2.circle(gray, (center_row, CHECKPOINT), 1, 90, 2)
    x1, y1 = center_row, CHECKPOINT
    for pt in gray[0, :]: 
            cv2.circle(gray, (x1, y1), 2, (0, 0, 255), 3)
    # cv2.imshow('Point', gray)
    center_of_road = int(math.sqrt(x1*x1+y1*y1))
    return center_of_road - 65

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

def segment(link):
    # if you want all classes
    # yolo_classes = list(model.names.values())
    # classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    conf = 0.8

    results = model.predict(link, conf=conf)
    # colors = [255,255,255]
    # print(results)
    # if results[0].masks is not None:
    #     for result in results:
    #         for mask, box in zip(result.masks.xy, result.boxes):
    #             points = np.int32([mask])
    #             # cv2.polylines(img, points, True, (255, 0, 0), 1)
    #             color_number = classes_ids.index(int(box.cls[0]))
    #             cv2.fillPoly(link, points, colors[color_number])
    #     cv2.imshow("Image", link)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Inference", annotated_frame)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # class index
            class_name = model.names[class_id]  # class name
            print(f"Detected class: {class_name}")
            return class_name
    return None

mode = 0
left = 0
stop = 0
right = 0
frame = 0
left_red = 0
right_red = 0

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            # crop = GetSeg()
            # crop = crop[90:180,:]
            # cv2.imshow('segment_image',segment_image)
            center_of_road = Midlane(segment_image)
            center_of_image = segment_image.shape[1] // 2
            deviation = center_of_road - (segment_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.52, i=0.02, d=0.12)
                # cv2.imshow('circle',circle)
            if segment(raw_image) == "TurnRight":
                    # cv2.imshow('Right',br)
                    print("Getting Ready To Turn Right")
                    mode = 3
            elif segment(raw_image) == "TurnLeft":
                    # cv2.imshow('Left',bl)
                    print("Getting Ready To Turn Left")
                    mode = 2
            elif segment(raw_image) == "NoTurnRight":
                    # cv2.imshow('Left',rr)
                    print("Getting Ready To Turn Left")
                    mode = 6
            elif segment(raw_image) == "NoTurnLeft":
                    # cv2.imshow('Right',rl)
                    print("Getting Ready To Turn Right")
                    mode = 5
            elif segment(raw_image) == "Stop":
                    # cv2.imshow('Stop',rs)
                    print("Getting Ready To Stop")
                    mode = 4
            else:
                    mode = 0
            # print(state)
            # resize = cv2.resize(GetRaw(), (800,800))
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            # cv2.imshow('crop', crop)

            # maxspeed = 90, max steering angle = 25
            if center_of_road == 95:
                center_of_road = 160
            if mode == 0:
                if left == 1 and frame >= 8:
                    AVControl( speed=30 , angle = -9 )
                    print("Turn Left Now")
                    if abs(deviation) >= 18:
                        # print(left)
                        print("Turn Left Done")
                        left = 0
                elif right == 1 and frame >= 8:
                    AVControl( speed=30 , angle = 9 )
                    print("Turn Right Now")
                    if abs(deviation) >= 18:
                        # print(right)
                        print("Turn Right Done")
                        right = 0
                elif stop == 1 and frame >= 8:
                    AVControl( speed=0 , angle = 0 )
                    print("Stop Now")
                    if abs(deviation) >= 18:
                        # print(stop)
                        print("Stop Done")
                        stop = 0
                elif left_red == 1 and frame >= 8:
                    AVControl( speed=30 , angle = -9 )
                    print("Turn Left Now")
                    if abs(deviation) >= 18:
                        # print(left)
                        print("Turn Left Done")
                        left_red = 0
                elif right_red == 1 and frame >= 8:
                    AVControl( speed=30 , angle = 9 )
                    print("Turn Right Now")
                    if abs(deviation) >= 18:
                        # print(right)
                        print("Turn Right Done")
                        right_red = 0
                else:
                    AVControl( speed=30 , angle = angle_setpoint )
                    frame = frame - 1
                    if frame < 0:
                        frame = 0
                    elif frame >= 10:
                        frame = 10
            # if mode == 1:
            #     for i in range(4000):
            #         AVControl( speed=30 , angle = 0 )
            #         t = t+1
            #     if t == 30:
            #         t = 0
            if mode == 2:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frame = frame + 1
                left = 1
            if mode == 3:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frame = frame + 1
                right = 1
            if mode == 4:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frame = frame + 1
                stop = 1
            if mode == 6:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frame = frame + 1
                left_red = 1
            if mode == 5:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frame = frame + 1
                right_red = 1
            # print("|  center_of_road = ",center_of_road)
            # print("|  center_of_image = ",center_of_image)
            print("|  deviation = ",deviation)
            print("|  frame = ",frame)
            # print("|  angle_setpoint = ",angle_setpoint)
            print("---------------------------------------")
            # print("Car x = ",segment_image.shape[1] // 2) #y=160
            # 0.3 0.01 0.05 Best condition 30
            #     0.06 0.14
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()