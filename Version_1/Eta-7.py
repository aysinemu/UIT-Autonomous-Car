from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import math

error_arr = np.zeros(5)
pre_t = time.time()

def handle_speed(straight_detected):
    if straight_detected:
        reduced_speed = 60
        set_motor_speed(reduced_speed)
    else:
        output = pidd.compute(current_speed)
        set_motor_speed(output)

def set_motor_speed(speed):
    global current_speed
    current_speed = speed

class PIDD:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp  
        self.Ki = Ki  
        self.Kd = Kd  
        self.setpoint = setpoint  
        self.prev_error = 0  
        self.integral = 0 
        self.max_output = 28  
    def compute(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error
        derivative = error - self.prev_error
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        output = np.clip(output, 0, self.max_output)
        return output

pidd = PIDD(Kp=900.0, Ki=100.0, Kd=120.0, setpoint=28)
current_speed = 0

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
    if int(deviation) >= -1.2 and int(deviation) <= 1.2 :
        straight_detected = True  
        handle_speed(straight_detected)
    else:
        straight_detected = False
    return int(angle)

CHECKPOINT = 160

def Midlane(image):
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
    # cv2.imshow('test', gray)
    x1, y1 = center_row, CHECKPOINT
    center_of_road = int(math.sqrt(x1*x1+y1*y1))
    return center_of_road - 65

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            # raw_image = GetRaw()
            segment_image = GetSeg()
            print(state)
            # resize = cv2.resize(GetRaw(), (800,800))
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            speed = int(state['Speed'])
            print(speed)
            control_output = pidd.compute(speed)
            set_motor_speed(control_output)
            # maxspeed = 90, max steering angle = 25
            center_of_road = Midlane(segment_image)
            center_of_image = segment_image.shape[1] // 2
            deviation = center_of_road - (segment_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.3, i=0.06, d=0.14)
            print("center_of_road = ",center_of_road)
            print("center_of_image = ",center_of_image)
            print("deviation = ",deviation)
            print("angle_setpoint = ",angle_setpoint)
            AVControl( speed = current_speed , angle = angle_setpoint )
            # 0.3 0.01 0.05 Best condition 30
            #     0.06 0.14
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
