from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time
import math
import asyncio

detect_circle_bool = 0 # 0 False 1 True
detect_blue_sign_bool = 0
detect_red_sign_bool = 0
detect_arrow_direction_bool = 0 
center_sign_x = 0
center_sign_y = 0 
car_y = 160
car_x = 160
angle_sign = 0

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

def get_filter_arrow_image(threslold_image):
    blank_image = np.zeros_like(threslold_image)

    # dilate image to remove self-intersections error
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    threslold_image = cv2.dilate(threslold_image, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(threslold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:

        threshold_distnace = 1000

        for cnt in contours:
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    start_index, end_index, farthest_index, distance = defects[i, 0]

                    # you can add more filteration based on this start, end and far point
                    # start = tuple(cnt[start_index][0])
                    # end = tuple(cnt[end_index][0])
                    # far = tuple(cnt[farthest_index][0])

                    if distance > threshold_distnace:
                        cv2.drawContours(blank_image, [cnt], -1, 255, -1)

        return blank_image
    else:
        return None


def get_length(p1, p2):
    line_length = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return line_length


def get_max_distace_point(cnt):
    max_distance = 0
    max_points = None
    for [[x1, y1]] in cnt:
        for [[x2, y2]] in cnt:
            distance = get_length((x1, y1), (x2, y2))

            if distance > max_distance:
                max_distance = distance
                max_points = [(x1, y1), (x2, y2)]

    return max_points


def angle_beween_points(a, b):
    arrow_slope = (a[0] - b[0]) / (a[1] - b[1])
    arrow_angle = math.degrees(math.atan(arrow_slope))
    return arrow_angle


def get_arrow_info(arrow_image):
    global angle_sign
    arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrow_info = []
    if hierarchy is not None:

        for cnt in contours:
            # draw single arrow on blank image
            blank_image = np.zeros_like(arrow_image)
            cv2.drawContours(blank_image, [cnt], -1, 255, -1)

            point1, point2 = get_max_distace_point(cnt)

            angle = angle_beween_points(point1, point2)
            lenght = get_length(point1, point2)
            angle_sign = angle
            print("Arrow Sign Angle:",angle)
            print("Arrow Sign Length:",lenght)

            cv2.line(arrow_info_image, point1, point2, (0, 255, 255), 1)

            cv2.circle(arrow_info_image, point1, 2, (255, 0, 0), 3)
            cv2.circle(arrow_info_image, point2, 2, (255, 0, 0), 3)

            cv2.putText(arrow_info_image, "angle : {0:0.2f}".format(angle),
                        point2, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
            cv2.putText(arrow_info_image, "lenght : {0:0.2f}".format(lenght),
                        (point2[0], point2[1] + 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)

        return arrow_info_image, arrow_info
    else:
        return None, None

def red_sign(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([155,25,0])
    upper_blue = np.array([179,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    red_ratio = np.sum(mask) / (mask.size * 255)
    global detect_red_sign_bool
    if red_ratio > 0.001:
        detect_red_sign_bool = 1
        return res
    else:
        detect_red_sign_bool = 0
        return None

def blue_sign(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    lower_blue = np.array([110,50,50]) 
    upper_blue = np.array([130,255,255]) 
  
    # Here we are defining range of bluecolor in HSV 
    # This creates a mask of blue coloured  
    # objects found in the frame. 
    mask = cv2.inRange(hsv, lower_blue, upper_blue) 
  
    # The bitwise and of the frame and mask is done so  
    # that only the blue coloured objects are highlighted  
    # and stored in res 
    res = cv2.bitwise_and(image,image, mask= mask) 
    blue_ratio = np.sum(mask) / (mask.size * 255)
    global detect_blue_sign_bool
    if blue_ratio > 0.005:
        detect_blue_sign_bool = 1
        return res
    else:
        detect_blue_sign_bool = 0
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
	vertices = np.array([[(0, 180), (320, 180), (320, 80), (0, 80)]], dtype=np.int32)
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
        # for pt in detected_circles[0, :]: 
        #     cv2.circle(result, (a, b), 1, (0, 0, 255), 3)
        # print('Sign Center x=',a)
        # print('Sign Center y=',b)
        detect_circle_bool = 1
        center_sign_x = a
        center_sign_y = b
        return result
    detect_circle_bool = 0
    return None

def arrow(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    arrow_image = get_filter_arrow_image(thresh_image)
    if arrow_image is not None:
        arrow_info_image, arrow_info = get_arrow_info(arrow_image)
        return arrow_info_image , arrow_info

if __name__ == "__main__":
    try:
        while True:
            mode = 0
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            # crop = GetSeg()
            # crop = crop[90:180,:]
            circle = draw_circle(raw_image)
            if detect_circle_bool == 1:
                # distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                # print("distance_to_sign",distance_to_sign)
                blue = blue_sign(circle)
                # cv2.imshow("Detected Blue",blue)
                if detect_blue_sign_bool == 1:
                    arrow_dir = arrow(circle)
                    distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                    print("distance_to_sign",distance_to_sign)
                    print("Sign Blue")
                    print("Arrow Blue")
                    if angle_sign > 10:
                        print("Turning Left Mode")
                        mode = 2
                    elif angle_sign < -10:
                        print("Turning Right Mode")
                        mode = 3
                    else:
                        print("Forward Mode")
                        mode = 1 
                    # cv2.imshow("Detected Arrow Dir",arrow_dir)
                elif detect_blue_sign_bool == 0:
                    red = red_sign(circle)
                    if detect_red_sign_bool == 1:
                        arrow_dir = arrow(circle)
                        distance_to_sign = int(math.sqrt(pow((center_sign_x - car_x),2) + pow((center_sign_y - car_y),2)))
                        print("distance_to_sign",distance_to_sign)
                        print("Sign Red")
                        print("Arrow Red")
                        if angle_sign <= -40 and -50 >= angle_sign:
                            print("Turning Left Mode")
                            mode = 2
                        elif -50 >= angle_sign:
                            print("Turning Right Mode")
                            mode = 3
                    # arrow_dir = arrow(circle)
                    # print("Sign Red")
                    # print("Arrow Red")
                    # # cv2.imshow("Detected Arrow Dir",arrow_dir)
            else:
                mode == 0        
            # print(state)
            # resize = cv2.resize(GetRaw(), (800,800))
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            # cv2.imshow('crop', crop)

            # maxspeed = 90, max steering angle = 25
            if mode == 0:
                center_of_road = Midlane(segment_image)
                center_of_image = segment_image.shape[1] // 2
                deviation = center_of_road - (segment_image.shape[1] // 2)
                angle_setpoint = PID(deviation, p=0.3, i=0.03, d=0.05)
                AVControl( speed=30 , angle = angle_setpoint )
            if mode == 1:
                    AVControl( speed=30 , angle = 0 )
            if mode == 2:
                    AVControl( speed=30 , angle = -18 )
            if mode == 3:
                    AVControl( speed=30 , angle = 18 )
            print("|  center_of_road = ",center_of_road)
            print("|  center_of_image = ",center_of_image)
            print("|  deviation = ",deviation)
            print("|  angle_setpoint = ",angle_setpoint)
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
