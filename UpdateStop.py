from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import math

detect_circle_bool = 0 # 0 False 1 True
detect_blue_right_sign_bool = 0
detect_blue_left_sign_bool = 0
detect_red_right_sign_bool = 0
detect_red_stop_sign_bool = 0
detect_red_left_sign_bool = 0
detect_arrow_direction_bool = 0 
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

def stop_red_sign(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([90,90,90])
    upper_blue = np.array([90,90,90])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    red_ratio = np.sum(mask) / (mask.size * 255)
    global detect_red_stop_sign_bool
    if red_ratio > 0.0002:
        detect_red_stop_sign_bool = 1
        return res
    else:
        detect_red_stop_sign_bool = 0
        return None


def left_red_sign(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([179,255,179])
    upper_blue = np.array([179,255,179])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    red_ratio = np.sum(mask) / (mask.size * 255)
    global detect_red_left_sign_bool
    if red_ratio > 0.0002:
        detect_red_left_sign_bool = 1
        return res
    else:
        detect_red_left_sign_bool = 0
        return None

def right_red_sign(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([128,128,128])
    upper_blue = np.array([128,128,128])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    red_ratio = np.sum(mask) / (mask.size * 255)
    global detect_red_right_sign_bool
    if red_ratio > 0.0002:
        detect_red_right_sign_bool = 1
        return res
    else:
        detect_red_right_sign_bool = 0
        return None

def left_blue_sign(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([255,255,1])
    upper_blue = np.array([255,255,1])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    red_ratio = np.sum(mask) / (mask.size * 255)
    global detect_blue_left_sign_bool
    if red_ratio > 0.0002:
        detect_blue_left_sign_bool = 1
        return res
    else:
        detect_blue_left_sign_bool = 0
        return None

def right_blue_sign(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    lower_blue = np.array([179,179,255]) 
    upper_blue = np.array([179,179,255]) 
  
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(image, lower_blue, upper_blue) 
  
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(image,image, mask= mask) 
    blue_ratio = np.sum(mask) / (mask.size * 255)
    global detect_blue_right_sign_bool
    if blue_ratio > 0.0002:
        detect_blue_right_sign_bool = 1
        return res
    else:
        detect_blue_right_sign_bool = 0
        return None
    # return res

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

def region_selection(image):
	mask = np.zeros_like(image) 
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
    # Bottom-left corner: (230, 80)
    # Bottom-right corner: (320, 80)
    # Top-right corner: (320, 0)
    # Top-left corner: (230, 0)
	vertices = np.array([[(180, 130), (320, 130), (320, 0), (180, 0)]], dtype=np.int32)
	# vertices = np.array([[(230, 80), (320, 80), (320, 0), (230, 0)]], dtype=np.int32)
	masked_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def draw_circle(image):
    global detect_circle_bool
    global center_sign_x
    global center_sign_y
    global detect_circle_bool
    roi = region_selection(image)
    image = apply_roi(image , roi)
    # cv2.imshow("Mask",image)
    # Convert to grayscale. 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 

    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred, 
                    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                param2 = 30, minRadius = 1, maxRadius = 40) 
    mask = np.zeros_like(image)
    # Draw circles that are detected. 
    if detected_circles is not None: 

        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 

        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 

            # Draw the circumference of the circle. 
            cv2.circle(mask, (a, b), r, (255, 255, 255), -1)  

            # Draw a small circle (of radius 1) to show the center. 
            # cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        result = cv2.bitwise_and(image, image, mask=mask_gray)
        for pt in detected_circles[0, :]: 
            cv2.circle(result, (a, b), 1, (0, 0, 255), 3)
        # print('Sign Center x=',a)
        # print('Sign Center y=',b)
        detect_circle_bool = 1
        center_sign_x = a
        center_sign_y = b
        return result
    detect_circle_bool = 0
    return None

mode = 0
left = 0
stop = 0
right = 0

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            # crop = GetSeg()
            # crop = crop[90:180,:]
            # cv2.imshow('segment_image',segment_image)
            circle = draw_circle(raw_image)
            if detect_circle_bool == 1:
                # cv2.imshow('circle',circle)
                br = right_blue_sign(segment_image)
                bl = left_blue_sign(segment_image)
                rr = right_red_sign(segment_image)
                rl = left_red_sign(segment_image)
                rs = stop_red_sign(segment_image)
                if detect_blue_right_sign_bool == 1:
                    distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                    print("distance_to_sign",distance_to_sign)
                    cv2.imshow('Right',br)
                    print("Turn Right")
                    mode = 3
                elif detect_blue_left_sign_bool == 1:
                    distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                    print("distance_to_sign",distance_to_sign)
                    cv2.imshow('Left',bl)
                    print("Turn Left")
                    mode = 2
                elif detect_red_right_sign_bool == 1:
                    distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                    print("distance_to_sign",distance_to_sign)
                    cv2.imshow('Left',rr)
                    print("Turn Left")
                    mode = 2
                elif detect_red_left_sign_bool == 1:
                    distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                    print("distance_to_sign",distance_to_sign)
                    cv2.imshow('Right',rl)
                    print("Turn Right")
                    mode = 3
                elif detect_red_stop_sign_bool == 1:
                    distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                    print("distance_to_sign",distance_to_sign)
                    cv2.imshow('Stop',rs)
                    print("Stop")
                    mode = 4
            else:
                mode = 0
            print(state)
            # resize = cv2.resize(GetRaw(), (800,800))
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            # cv2.imshow('crop', crop)

            # maxspeed = 90, max steering angle = 25
            center_of_road = Midlane(segment_image)
            if center_of_road == 95:
                center_of_road = 160
            center_of_image = segment_image.shape[1] // 2
            deviation = center_of_road - (segment_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.5, i=0.02, d=0.11)
            if mode == 0:
                if left == 1:
                    AVControl( speed=30 , angle = -9 )
                    if abs(deviation) > 20:
                        print(left)
                        left = 0
                elif right == 1:
                    AVControl( speed=30 , angle = 9 )
                    if abs(deviation) > 20:
                        print(right)
                        right = 0
                elif stop == 1:
                    AVControl( speed=0 , angle = 0 )
                    if abs(deviation) > 20:
                        print(stop)
                        stop = 0
                else:
                    AVControl( speed=30 , angle = angle_setpoint )
            # if mode == 1:
            #     for i in range(4000):
            #         AVControl( speed=30 , angle = 0 )
            #         t = t+1
            #     if t == 30:
            #         t = 0
            if mode == 2:
                pidd = PIDD(Kp=2.2, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                left = 1
            if mode == 3:
                pidd = PIDD(Kp=2.2, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                right = 1
            if mode == 4:
                pidd = PIDD(Kp=2.2, Ki=0, Kd=0, setpoint=10)
                speed = int(state['Speed'])
                control_output = pidd.compute(speed)
                set_motor_speed(control_output)
                AVControl( speed=current_speed , angle = angle_setpoint )
                stop = 1
            # print("|  center_of_road = ",center_of_road)
            # print("|  center_of_image = ",center_of_image)
            print("|  deviation = ",deviation)
            # print("|  angle_setpoint = ",angle_setpoint)
            # print("---------------------------------------")
            # print("Car x = ",segment_image.shape[1] // 2) #y=160
            # 0.3 0.01 0.05 Best condition 30
            #     0.06 0.14
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
