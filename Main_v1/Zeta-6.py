import cv2
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
from ultralytics import YOLO
from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import time

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


model_yolo_name='best_28.pt'
model_yolo = YOLO(model_yolo_name)
model_yolo.conf = 0.6

angle = 0
speed = 160
count = 0
No_Detect = 0

Move_Right = 0
Ready_Mode_Right = 0
Start_Turn_Right = 0

Move_Left = 0
Ready_Mode_Left = 0
Start_Turn_Left = 0

Ready_Mode_Straight = 0

arr_to_decide = []

def move_right(img):
    global Move_Right, Ready_Mode_Right, angle_setpoint,   Start_Turn_Right

    avg_area_top = np.mean(img[:20,140:160])
    avg_area_down = np.mean(img[30:70,140:160])
    
    # line_decided = np.mean(img[:,159])
    
    if avg_area_down > 250 and avg_area_top <= 10:
        Move_Right = 1
    if Move_Right:
        angle_setpoint = 18
    if avg_area_down <= 180 and Move_Right:
        print("Stoppppppppppppppppppppppppppppppppppppppppp")
        Move_Right = 0
        Start_Turn_Right = 0
    print("Run mode Right ###########################################################", angle)

    return angle_setpoint, Move_Right

def move_left(img):
    global Move_Left, Ready_Mode_Left, angle_setpoint,  Start_Turn_Left

    avg_area_top = np.mean(img[:20,:20])
    avg_area_down = np.mean(img[30:70,:20])
    
    # line_decided = np.mean(img[:,159])

    if avg_area_down > 250 and avg_area_top <= 10:
        Move_Left = 1
    if Move_Left:
        print("Goooooooooooooooooooooooooooooooooooooo")
        angle_setpoint = -18
    if avg_area_down <= 180 and Move_Left:
        print("Stoppppppppppppppppppppppppppppppppppppppppp")
        Move_Left = 0
        Start_Turn_Left = 0
    print("Run mode Left ###########################################################", avg_area_top)

    return angle_setpoint, Move_Left

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            print(state)
            # resize = cv2.resize(GetRaw(), (800,800))
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            # speed = int(state['Speed'])
            # print(speed)
            control_output = pidd.compute(speed)
            set_motor_speed(control_output)
            # maxspeed = 90, max steering angle = 25
            # 0.3 0.01 0.05 Best condition 30
            #     0.06 0.14
            results = model_yolo(raw_image, conf = 0.7)
            # {0: 'right', 1: 'left', 2: 'straight', 3: 'no_left', 4: 'no_right'}
            # print("Results", results)
            # for r in results:
            #     print("Result",r.boxes.xyxy )
            # print("Result", results)
            for result in results:
                try: 
                    xB = int(result.boxes.xyxy.cpu().detach().numpy()[0][2])
                    xA = int(result.boxes.xyxy.cpu().detach().numpy()[0][0])
                    yB = int(result.boxes.xyxy.cpu().detach().numpy()[0][3])
                    yA = int(result.boxes.xyxy.cpu().detach().numpy()[0][1])
                    name_img = result.boxes.cls.cpu().detach().numpy()[0]
                    area = (xB-xA)*(yB-yA)
                    print("Area equal: ", area)
                    if area >= 2000 and not (Ready_Mode_Right or Ready_Mode_Left or Ready_Mode_Straight):
                        arr_to_decide.append(name_img)
                    if len(arr_to_decide) >=5:
                        counter = Counter(arr_to_decide)
                        most_common_value = counter.most_common(1)[0][0]
                        print("Decidedddddddddddddddddddddddddd", most_common_value)
                        arr_to_decide = []
                        if most_common_value == 2:
                            Ready_Mode_Straight = 1
                            print("Detect straiggggggggggggggggggggggggggggggg")
                        elif most_common_value == 1 or most_common_value == 4:
                            Ready_Mode_Left = 1

                        elif most_common_value == 0 or most_common_value == 3:
                            Ready_Mode_Right = 1
                    print('result_detect:', name_img)
                    print('Area of box: ',area)
                    cv2.rectangle(raw_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                except:
                    continue
            cv2.imshow("IMG_BF", raw_image)

            img = raw_image[200:,:,:]

            img = Midlane(img)
            
            if Ready_Mode_Right == 0 and Ready_Mode_Left == 0:
                current_speed = 60
            else:
                print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRrrr")
                speed = 30

            # ##################################### PID SPEED ###########################################
            # err_speed = speed - current_speed
            # speed = PID(err_speed, 25, 1.5, 0, mode = "speed")

            # ###########################################################################################

            # Kp,Ki,Kd = 0.9, 1.7, 0
            arr = []
            lineRowAB = img[50:80,:]
            lineRowBL = img[50,:]
            for x,y in enumerate(lineRowBL):
                if y == 255:
                    arr.append(x)
            arrmax = max(arr)
            arrmin = min(arr)
            center=int((arrmax+arrmin)/2)
            # angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-50)))

            if (Ready_Mode_Right or Ready_Mode_Left or Ready_Mode_Straight) and name_img is None:
                No_Detect += 1
                if No_Detect >= 15 and Ready_Mode_Right:
                    Start_Turn_Right = 1
                    No_Detect = 0
                    Ready_Mode_Right = 0
                elif No_Detect >= 15 and Ready_Mode_Left:
                    Start_Turn_Left = 1
                    No_Detect = 0
                    Ready_Mode_Left = 0

                elif No_Detect >= 1 and No_Detect <=25 and Ready_Mode_Straight:
                    current_speed = 50
                    angle_setpoint = 0
                elif No_Detect >= 25 and Ready_Mode_Straight:
                    Start_Straight = 1
                    No_Detect = 0
                    Ready_Mode_Straight = 0
            if Start_Turn_Right:
                angle_setpoint,  Move_Right = move_right(img)
            elif Start_Turn_Left:
                angle_setpoint,  Move_Left = move_left(img)

            # ################################ PID Angle ######################################
            # if not (Move_Right or Move_Left or Ready_Mode_Straight):
            #     err = angle - current_angle
            #     angle = PID(err,Kp, Ki, Kd, mode = "speed") #PID(err, Kp, Ki, Kd)
            # #################################################################################

            center_of_road = Midlane(segment_image)
            center_of_image = segment_image.shape[1] // 2
            deviation = center_of_road - (segment_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.3, i=0.06, d=0.14)
            # print("center_of_road = ",center_of_road)
            # print("center_of_image = ",center_of_image)
            # print("deviation = ",deviation)
            # print("angle_setpoint = ",angle_setpoint)
            AVControl( speed = current_speed , angle = angle_setpoint )

            cv2.circle(img,(arrmin,70),5,(0,255,0),5)
            cv2.circle(img,(arrmax,70),5,(0,255,0),5)
            cv2.line(img,(center,50),(int(img.shape[1]/2),img.shape[0]),(0,0,255),(5))

            cv2.imshow("IMG", img)

            key = cv2.waitKey(1)

    finally:
        print('closing socket')
        CloseSocket()

