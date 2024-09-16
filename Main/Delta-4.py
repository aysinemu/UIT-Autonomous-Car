from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time 

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
    if delta_t != 0:
        D = (error - error_arr[1]) / delta_t * d
    else:
        D = 0
    I = np.sum(error_arr) * delta_t * i
    angle = P + I + D
    if abs(angle) > 25:
        angle = np.sign(angle) * 25
    return int(angle)

def remove_shadow(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([180, 255, 40])
    shadow_mask = cv2.inRange(hsv_image, lower_shadow, upper_shadow)
    shadow_mask_inv = cv2.bitwise_not(shadow_mask)
    image_no_shadow = cv2.bitwise_and(image, image, mask=shadow_mask_inv)
    return image_no_shadow

def region_selection(image):
	mask = np.zeros_like(image) 
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
	vertices = np.array([[(100,175), (220,175), (160,120)]], dtype=np.int32)
	masked_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def detect_white_line(image):
    image_without_shadow = remove_shadow(image)
    # cv2.imshow('image_without_shadow',image_without_shadow)
    image_without_shadow = region_selection(image_without_shadow)
    cv2.imshow('region_selection',image_without_shadow)
    gray_image = cv2.cvtColor(image_without_shadow, cv2.COLOR_BGR2GRAY)
    _, binary_white = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else image.shape[1] // 2
        return center_x + 4
    else:
        return image.shape[1] // 2

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            # segment_image = GetSeg()
            print(state)
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)

            # maxspeed = 90, max steering angle = 25
            center_of_road = detect_white_line(raw_image)
            center_of_image = raw_image.shape[1] // 2
            deviation = center_of_road - (raw_image.shape[1] // 2)
            angle_setpoint = PID(deviation, p=0.35, i=0.02, d=0.04)
            print("center_of_road = ",center_of_road)
            print("center_of_image = ",center_of_image)
            print("deviation = ",deviation)
            print("angle_setpoint = ",angle_setpoint)
            AVControl( speed=20 , angle = angle_setpoint )            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
