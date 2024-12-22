from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

model = YOLO("/workspace/best.pt")
# model = YOLO("/workspace/trainnew/train/weights/best.pt")
# model = YOLO("/workspace/SignDetect/weights/best.pt")

detect_circle_bool = 0 # 0 False 1 True
center_sign_x = 0
center_sign_y = 0 
car_y = 160
car_x = 160
current_speed = 0

error_arr = np.zeros(5)
pre_t = time.time()
# MAX_SPEED = 60

def sign(image):
    lostop = np.array([89,89,89])
    upstop = np.array([89,89,89])
    lostop1 = np.array([90,90,90])
    upstop1 = np.array([90,90,90])
    lolered = np.array([179,255,179])
    uplered = np.array([179,255,179])
    lorired = np.array([128,128,128])
    uprired = np.array([128,128,128])
    lolebl = np.array([255,255,1])
    uplebl = np.array([255,255,1])
    loribl = np.array([179,178,255]) 
    upribl = np.array([179,178,255]) 
    loribl1 = np.array([179,179,255])
    upribl1 = np.array([179,179,255])  
    mask = cv2.inRange(image, lostop, upstop)
    mask1 = cv2.inRange(image, lostop1, upstop1)
    mask2 = cv2.inRange(image, lolered, uplered)
    mask3 = cv2.inRange(image, lorired, uprired)
    mask4 = cv2.inRange(image, lolebl, uplebl)
    mask5 = cv2.inRange(image, loribl, upribl)
    mask6 = cv2.inRange(image, loribl1, upribl1)
    res = cv2.bitwise_and(image,image, mask= mask)
    res1 = cv2.bitwise_and(image,image, mask= mask1)
    res2 = cv2.bitwise_and(image,image, mask= mask2)
    res3 = cv2.bitwise_and(image,image, mask= mask3)
    res4 = cv2.bitwise_and(image,image, mask= mask4)
    res5 = cv2.bitwise_and(image,image, mask= mask5)
    res6 = cv2.bitwise_and(image,image, mask= mask6)
    sign = np.sum(mask) / (mask.size * 255)
    sign1 = np.sum(mask1) / (mask1.size * 255)
    sign2 = np.sum(mask2) / (mask2.size * 255)
    sign3 = np.sum(mask3) / (mask3.size * 255)
    sign4 = np.sum(mask4) / (mask4.size * 255)
    sign5 = np.sum(mask5) / (mask5.size * 255)
    sign6 = np.sum(mask6) / (mask6.size * 255)
    global detect_circle_bool
    if sign > 0.00002:
        detect_circle_bool = 1
    elif sign1 > 0.00002:
        detect_circle_bool = 1
    elif sign2 > 0.00002:
        detect_circle_bool = 1
    elif sign3 > 0.00002:
        detect_circle_bool = 1
    elif sign4 > 0.00002:
        detect_circle_bool = 1
    elif sign5 > 0.00002:
        detect_circle_bool = 1
    elif sign6 > 0.00002:
        detect_circle_bool = 1
    else:
        detect_circle_bool = 0

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
    if delta_t != 0:
        D = (error - error_arr[1]) / delta_t * d
    else:
        D = 0
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 25:
        angle = np.sign(angle) * 25
    return int(angle)

CHECKPOINT = 160
SECONDPOINT = 100

def SecondPoint(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray*(255/np.max(gray))).astype(np.uint8)
    # cv2.imshow("test", gray)
    h, w = gray.shape
    line_row = gray[SECONDPOINT, :]
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
    center_roww = int((max_x+min_x)/2)
    gray = cv2.circle(gray, (center_roww, SECONDPOINT), 1, 90, 2)
    # cv2.imshow('test', gray)
    x1, y1 = center_roww, SECONDPOINT
    center_of_roadd = int(math.sqrt(x1*x1+y1*y1))
    return center_of_roadd - 65

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
    roi = cv2.resize(src=roi, dsize=(img.shape[1], img.shape[0]))
    
    assert img.shape[:2] == roi.shape[:2]
    
    thresh, roi = cv2.threshold(roi, thresh=128, maxval=1, type=cv2.THRESH_BINARY)
    
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
	vertices = np.array([[(0, 180), (320, 180), (320, 159), (0, 159)]], dtype=np.int32)
    # vertices = np.array([[(230, 80), (320, 80), (320, 0), (230, 0)]], dtype=np.int32)
	masked_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def segment(link):
    # if you want all classes
    # yolo_classes = list(model.names.values())
    # classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    conf = 0.3

    results = model.predict(link, conf=conf, verbose=False)
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
    # cv2.imshow("YOLO Inference", annotated_frame)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # class index
            class_name = model.names[class_id]  # class name
            # print(f"Detected class: {class_name}")
            return class_name
    return None

mode = 0
left = 0
stop = 0
right = 0
forward = 0
frameright = 0
frameleft = 0
framestop = 0
frameredright = 0
frameredleft = 0
left_red = 0
right_red = 0
yolo = 0
turn = 16.3
# turn = 10.2
devi = 12
timer = 0
speed_for_turn = 17

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            sign(segment_image)
            secondpoint = SecondPoint(segment_image)
            center_of_road = Midlane(segment_image)
            center_of_image = segment_image.shape[1] // 2
            deviation = center_of_road - (segment_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.52, i=0.02, d=0.12)
            if detect_circle_bool == 1:
                yolo = 1
            if yolo == 1:
                if segment(raw_image) == "TurnRight":
                    # print("Getting Ready To Turn Right")
                    mode = 3
                elif segment(raw_image) == "TurnLeft":
                    # print("Getting Ready To Turn Left")
                    mode = 2
                elif segment(raw_image) == "NoTurnRight":
                    # print("Getting Ready To Turn Left")
                    mode = 6
                elif segment(raw_image) == "NoTurnLeft":
                    # print("Getting Ready To Turn Right")
                    mode = 5
                elif segment(raw_image) == "Stop":
                    # print("Getting Ready To Stop")
                    mode = 4
                elif segment(raw_image) == None:
                    yolo = 0
            else:
                mode = 0
            if center_of_road == 95:
                center_of_road = 160
            if secondpoint == 35:
                forward = 1
            if mode == 0:
                if frameleft > (frameredright+frameredleft+frameright+framestop) and frameleft >= 3:
                    AVControl( speed=speed_for_turn , angle = -turn )
                    # print("Turn Left Now")
                    if abs(deviation) >= devi:
                        print("Turn Left Done")
                        AVControl( speed=30 , angle = angle_setpoint ) 
                        # framestop = 0
                        # frameright = 0
                        # frameredleft = 0
                        # frameredright = 0
                        frameleft = 0
                elif frameright > (frameredright+frameredleft+frameleft+framestop) and frameright >= 3:
                    AVControl( speed=speed_for_turn , angle = turn )
                    # print("Turn Right Now")
                    if abs(deviation) >= devi:
                        print("Turn Right Done")
                        AVControl( speed=30 , angle = angle_setpoint ) 
                        # framestop = 0
                        # frameleft = 0
                        # frameredleft = 0
                        # frameredright = 0
                        frameright = 0
                elif framestop > (frameredright+frameredleft+frameright+frameleft):
                    # AVControl( speed=0 , angle = 0 )
                    # print("Stop Now")
                    # timer = timer + 1
                    # if abs(timer) >= 10:
                    #     print("Stop Done")
                    # frameright = 0
                    # frameleft = 0
                    # frameredleft = 0
                    # frameredright = 0
                    framestop = 0
                elif frameredleft > (frameredright+frameleft+frameright+framestop) and frameredleft >= 3:
                    if forward == 1:
                        AVControl( speed=speed_for_turn , angle = -turn )
                        # print("Turn Left Now")
                        if abs(deviation) >= devi:
                            print("Turn Left Done")
                            AVControl( speed=30 , angle = angle_setpoint ) 
                            # frameright = 0
                            # frameleft = 0
                            # framestop = 0
                            # frameredright = 0
                            frameredleft = 0
                            forward = 0
                    else:
                        # frameright = 0
                        # frameleft = 0
                        # framestop = 0
                        # frameredright = 0
                        frameredleft = 0
                elif frameredright > (frameright+frameredleft+frameright+framestop) and frameredright >= 3:
                    if forward == 1:
                        AVControl( speed=speed_for_turn , angle = turn )
                        # print("Turn Right Now")
                        if abs(deviation) >= devi:
                            print("Turn Right Done")
                            AVControl( speed=30 , angle = angle_setpoint ) 
                            # frameright = 0
                            # frameleft = 0
                            # framestop = 0
                            # frameredleft = 0
                            frameredright = 0
                            forward = 0
                    else:
                        # frameright = 0
                        # frameleft = 0
                        # framestop = 0
                        # frameredleft = 0
                        frameredright = 0
                else:
                    AVControl( speed=20 , angle = angle_setpoint ) 
                    # AVControl( speed=30 , angle = angle_setpoint )     
                    forward = 0    
            if mode == 2:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frameleft = frameleft + 1
                forward = 0
                # left = 1
            if mode == 3:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frameright = frameright + 1
                forward = 0
                # right = 1
            if mode == 4:
                timer = timer + 1
                if abs(timer) >= 30:
                    pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                else:
                    pidd = PIDD(Kp=10000, Ki=10000, Kd=10000, setpoint=0)
                    print("Đang dừng lại 10s nha BTC =))) thấy stop sớm nên dừng sớm cho chắc :>>>")
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                framestop = framestop + 1
                forward = 0
                # stop = 1
            if mode == 6:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frameredleft = frameredleft + 1
                forward = 0
                # left_red = 1
            if mode == 5:
                pidd = PIDD(Kp=1, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                frameredright = frameredright + 1
                forward = 0
                # right_red = 1
            # print("|  center_of_road = ",center_of_road)
            # print("|  center_of_image = ",center_of_image)
            print("|  deviation = ",deviation)
            print("|  frameleft = ",frameleft)
            print("|  frameright = ",frameright)
            print("|  framestop = ",framestop)
            print("|  frameredleft = ",frameredleft)
            print("|  frameredright = ",frameredright)
            print("|  yolo = ",yolo)
            print("|  timer = ",timer)
            print("|  forward = ",forward)
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