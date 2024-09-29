import socket       
import cv2
import numpy as np
import json
import base64
import pandas 

from collections import Counter
import math
import torch
from torchvision.transforms import ToTensor

import time
import matplotlib.pyplot as plt
from line import *

from line_detect import process_img

from pid import PID
from ultralytics import YOLO


# Create a socket object 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

# Define the port on which you want to connect 
port = 54321                

# connect to the server on local computer 
s.connect(('127.0.0.1', port)) 



import torch
import os
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
    global Move_Right, Ready_Mode_Right, angle,   Start_Turn_Right

    avg_area_top = np.mean(img[:20,140:160])
    avg_area_down = np.mean(img[30:70,140:160])
    
    # line_decided = np.mean(img[:,159])
    
    if avg_area_down > 250 and avg_area_top <= 10:
        Move_Right = 1
    if Move_Right:
        angle = 18
    if avg_area_down <= 180 and Move_Right:
        print("Stoppppppppppppppppppppppppppppppppppppppppp")
        Move_Right = 0
        Start_Turn_Right = 0
    print("Run mode Right ###########################################################", angle)

    return angle, Move_Right

def move_left(img):
    global Move_Left, Ready_Mode_Left, angle,  Start_Turn_Left

    avg_area_top = np.mean(img[:20,:20])
    avg_area_down = np.mean(img[30:70,:20])
    
    # line_decided = np.mean(img[:,159])

    if avg_area_down > 250 and avg_area_top <= 10:
        Move_Left = 1
    if Move_Left:
        print("Goooooooooooooooooooooooooooooooooooooo")
        angle = -18
    if avg_area_down <= 180 and Move_Left:
        print("Stoppppppppppppppppppppppppppppppppppppppppp")
        Move_Left = 0
        Start_Turn_Left = 0
    print("Run mode Left ###########################################################", avg_area_top)

    return angle, Move_Left

    


if __name__ == "__main__":
    try:
        """
            - Chương trình đưa cho bạn 3 giá trị đầu vào:
                * image: hình ảnh trả về từ xe
                * current_speed: vận tốc hiện tại của xe
                * current_angle: góc bẻ lái hiện tại của xe
            - Bạn phải dựa vào giá trị đầu vào này để tính toán và
            gán lại góc lái và tốc độ xe vào 2 biến:
                * Biến điều khiển: sendBack_angle, sendBack_Speed
                Trong đó:
                    + sendBack_angle (góc điều khiển): [-25, 25]
                        NOTE: ( âm là góc trái, dương là góc phải)
                    + sendBack_Speed (tốc độ điều khiển): [0, 150]
            """
        while True:
            # Send data để điều khiển xe
            message = bytes(f"{angle} {speed}", "utf-8")
            s.sendall(message)

            # Recive data from server
            data = s.recv(100000)
            try:
                data_recv = json.loads(data)
            except:
                continue
            

            # Angle and speed recv from serverS
            current_angle = data_recv["Angle"]
            current_speed = data_recv["Speed"]
            print("angle: ", current_angle)
            print("speed: ", current_speed)
            print("---------------------------------------")
            #Img data recv from server
            jpg_original = base64.b64decode(data_recv["Img"])
            jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)

            name_img = None
            results = model_yolo(img, conf = 0.7)
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
                        

                    cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
                except:
                    continue
            cv2.imshow("IMG_BF", img)
                

            # Cut half of picture
            img = img[200:,:,:]

            img = process_img(img)
            
            if Ready_Mode_Right == 0 and Ready_Mode_Left == 0:
                speed = 50
            else:
                print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRrrr")
                speed = 35

            ##################################### PID SPEED ###########################################
            err_speed = speed - current_speed
            speed = PID(err_speed, 25, 1.5, 0, mode = "speed")

            ###########################################################################################

            Kp,Ki,Kd = 0.9, 1.7, 0
            arr = []
            lineRowAB = img[50:80,:]
            lineRowBL = img[50,:]
            for x,y in enumerate(lineRowBL):
                if y == 255:
                    arr.append(x)
            arrmax = max(arr)
            arrmin = min(arr)
            center=int((arrmax+arrmin)/2)
            angle = math.degrees(math.atan((center-img.shape[1]/2)/(img.shape[0]-50)))

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
                    speed = 70
                    angel = 0
                elif No_Detect >= 25 and Ready_Mode_Straight:
                    Start_Straight = 1
                    No_Detect = 0
                    Ready_Mode_Straight = 0
            if Start_Turn_Right:
                angle,  Move_Right = move_right(img)
            elif Start_Turn_Left:
                angle,  Move_Left = move_left(img)



            
            ################################ PID Angle ######################################
            if not (Move_Right or Move_Left or Ready_Mode_Straight):
                err = angle - current_angle
                angle = PID(err,Kp, Ki, Kd, mode = "speed") #PID(err, Kp, Ki, Kd)
            #################################################################################


            cv2.circle(img,(arrmin,70),5,(0,255,0),5)
            cv2.circle(img,(arrmax,70),5,(0,255,0),5)
            cv2.line(img,(center,50),(int(img.shape[1]/2),img.shape[0]),(0,0,255),(5))

            

            cv2.imshow("IMG", img)

            key = cv2.waitKey(1)
        
            

    finally:
        print('closing socket')
        s.close()

