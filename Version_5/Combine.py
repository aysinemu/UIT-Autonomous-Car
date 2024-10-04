from client_lib import GetStatus, GetRaw, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import math
import time

# PID Tuning parameters
CHECKPOINT_1 = 80
CHECKPOINT_2 = 60
CHECKPOINT_I = 28

pre_t = time.time()
err_arr = np.zeros(5)

Kp_vals = []
Ki_vals = []
Kd_vals = []
time_vals = []

# -----------------------------------------------------------------------------
def vector_length(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def AngCal(image):
    global lane_width_2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray * (255 / np.max(gray))).astype(np.uint8)
    _, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    h, w = bin.shape
    line_row_1 = bin[CHECKPOINT_1, :]
    line_row_2 = bin[CHECKPOINT_2, :]

    flag = True
    min_1, max_1 = 0, 0
    for x, y in enumerate(line_row_1):
        if y == 255 and flag:
            flag = False
            min_1 = x + 20
        elif y == 255:
            max_1 = x - 20

    flag = True
    min_2, max_2 = 0, 0
    for x, y in enumerate(line_row_2):
        if y == 255 and flag:
            flag = False
            min_2 = x
        elif y == 255:
            max_2 = x

    center_1 = (min_1 + max_1) // 2
    center_2 = (min_2 + max_2) // 2

    lane_width_1 = max_1 - min_1
    lane_width_2 = max_2 - min_2

    # Tính toán lỗi
    error = center_1 - w // 2

    # Vẽ đường biểu diễn hướng đi của xe
    cv2.line(image, (w//2, CHECKPOINT_1), (center_1, CHECKPOINT_1), (0, 255, 0), 2)
    cv2.line(image, (w//2, CHECKPOINT_2), (center_2, CHECKPOINT_2), (0, 255, 0), 2)

    return error

# Nhận diện ngã 3/ngã 4
def detect_intersection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    h, w = bin.shape

    # Kiểm tra hàng pixel ở dưới cùng (gần xe) để phát hiện các làn đường
    bottom_row = bin[h-20, :]

    left_lane = np.any(bottom_row[:w//3] == 255)
    center_lane = np.any(bottom_row[w//3:w*2//3] == 255)
    right_lane = np.any(bottom_row[w*2//3:] == 255)

    if left_lane and center_lane and right_lane:
        return "Ngã 4"
    elif (left_lane and center_lane) or (center_lane and right_lane):
        return "Ngã 3"
    else:
        return "Đường thẳng"

# Detect if a curve is happening based on error changes
def is_curve_detected(error, prev_error, threshold=20):
    error_change = abs(error - prev_error)
    return error_change > threshold

# PID control function that uses different gains for curves
def PID_control(err, Kp, Ki, Kd, Kp_curve, Ki_curve, Kd_curve, is_curve):
    global pre_t
    err_arr[1:] = err_arr[0:-1]
    err_arr[0] = err
    delta_t = time.time() - pre_t
    pre_t = time.time()

    if is_curve:
        P = Kp_curve * err
        I = Ki_curve * np.sum(err_arr) * delta_t
        D = Kd_curve * (err - err_arr[1]) / delta_t
    else:
        P = Kp * err
        I = Ki * np.sum(err_arr) * delta_t
        D = Kd * (err - err_arr[1]) / delta_t

    angle = P + I + D
    if abs(angle) > 25:
        angle = np.sign(angle) * 25
        
    return int(angle)

# Main loop to run the vehicle control system
if __name__ == "__main__":
    # PID parameters for straight path and curves
    Kp, Ki, Kd = 0.18, 0.0, 0.065  # For straight paths
    Kp_curve, Ki_curve, Kd_curve = 0.28, 0.0, 0.2  # For curves

    speed_max = 65  # Maximum speed
    speed_min = 35  # Minimum speed
    prev_error = 0
    curve_detected = False

    try:
        while True:
            state = GetStatus()
            print(state)
            segment_image = GetSeg()  # Get the segmented image
            
            # Crop and analyze the lane
            height, width, _ = segment_image.shape
            crop_seg = segment_image[50:height, 0:width]
            
            # Calculate the error from the lane
            error = AngCal(crop_seg)
            
            # Detect if the car is entering a curve
            curve_detected = is_curve_detected(error, prev_error)
            
            # Apply PID control based on curve detection
            angle = PID_control(error, Kp, Ki, Kd, Kp_curve, Ki_curve, Kd_curve, curve_detected)
            
            # Adjust speed based on the error
            A = 0.5
            speed = -A * abs(error) + speed_max
            speed = np.clip(speed, speed_min, speed_max)
            
            # Control the vehicle
            AVControl(speed=30, angle=angle)
            
            # Nhận diện ngã 3, ngã 4
            #intersection_type = detect_intersection(segment_image)
            #cv2.putText(segment_image, intersection_type, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Hiển thị ảnh phân đoạn có vẽ hướng đi và thông tin ngã 3/ngã 4
            cv2.imshow('Segment Image', segment_image)
            #In góc lái và speed
            print(f"Angle: {angle} |  Speed: {speed}")
            print(f"Kp: {Kp:2f}, Ki: {Ki:2f}, Kd: {Kd:2f}")
            print(f"-----------------------------")
            # Update previous error for the next loop
            prev_error = error
            
            # Exit on 'q' key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('Closing socket')
        CloseSocket()