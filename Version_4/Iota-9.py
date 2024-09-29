from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2
import numpy as np
import time 
import math

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
    if red_ratio > 0.002:
        print(red_ratio)
        return res
    else:
        return image
    # print(red_ratio)
    # return res

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
    if blue_ratio > 0.01:
        return res
    else:
        return image
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
    # Bottom-left corner: (230, 80) 180 130
    # Bottom-right corner: (320, 80) 320 130
    # Top-right corner: (320, 0)
    # Top-left corner: (230, 0)
	vertices = np.array([[(180, 130), (320, 130), (320, 0), (180, 0)]], dtype=np.int32)
    # vertices = np.array([[(230, 80), (320, 80), (320, 0), (230, 0)]], dtype=np.int32)
	masked_image = cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def draw_circle(image):
    roi = region_selection(image)
    image = apply_roi(image , roi)
    cv2.imshow("Mask",image)
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
        print('x=',a)
        print('y=',b)
        return result
    return image

def arrow(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    arrow_image = get_filter_arrow_image(thresh_image)
    if arrow_image is not None:
        arrow_info_image, arrow_info = get_arrow_info(arrow_image)
        return arrow_info_image

# def sign_color(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return gray

# # img = cv2.imread('/workspace/TurnRight/img_84.jpg', cv2.IMREAD_COLOR) #< -50 
# # img = cv2.imread('/workspace/TurnLeft/img_47.jpg', cv2.IMREAD_COLOR) #> 50
# # img = cv2.imread('/workspace/Black.png', cv2.IMREAD_COLOR) 
# img = cv2.imread('/workspace/NoTurnRight/img_78.jpg', cv2.IMREAD_COLOR) #> -40
# # img = cv2.imread('/workspace/NoTurnLeft/img_35.jpg', cv2.IMREAD_COLOR) #> -50
# roi_road = region_selection_road(img)
# image = apply_roi(img,roi_road)
# cv2.imshow("Road",image)
# circle = draw_circle(img)
# blue = blue_sign(circle)
# red = red_sign(circle)
# arrow_dir = arrow(circle)
# cv2.imshow("Detected Circle",circle)
# cv2.imshow("Detected Blue",blue)
# cv2.imshow("Detected Red",red)
# cv2.imshow("Detected Arrow Dir",arrow_dir)
# cv2.waitKey(0) 

def color_from_sign(image):
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([128,128,128])
    upper_blue = np.array([128,128,128])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image,image, mask= mask)
    ratio = np.sum(mask) / (mask.size * 255)
    # if ratio > 0.002:
    print(ratio)
    return res
    # else:
    # return image
    # print(red_ratio)
    # return res

if __name__ == "__main__":
    try:
        while True:
            # raw_image = GetRaw()
            segment_image = GetSeg()
            cv2.imshow('raw_image', segment_image)
            # segment_image = sign_color(segment_image)
            circle = draw_circle(segment_image)
            cv2.imshow('draw_circle',circle)
            color = color_from_sign(circle)
            cv2.imshow('color_from_sign',color)
            #Right 255 179 179 0.023368055555555555
            #Left 1 255 255 
            #NL 179 255 179
            #NR 128 128 128
            # cv2.imshow('segment_image', circle)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
