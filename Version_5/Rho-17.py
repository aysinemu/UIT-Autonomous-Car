from ultralytics import YOLO
import random
import cv2
import numpy as np
import math

cl = 40

model = YOLO("/workspace/RoadSeg/weights/best.pt")

def blue_road(image):
    lower_blue = np.array([cl,0,0]) 
    upper_blue = np.array([cl,0,0]) 
    mask = cv2.inRange(image, lower_blue, upper_blue) 
    res = cv2.bitwise_and(image,image, mask= mask) 
    return res

CHECKPOINT = 130

def segment(link):
    # if you want all classes
    link = cv2.imread(link)
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.8

    results = model.predict(link, conf=conf)
    colors = [cl,cl,cl]
    # print(results)
    if results[0].masks is not None:
        for result in results:
            for mask, box in zip(result.masks.xy, result.boxes):
                points = np.int32([mask])
                # cv2.polylines(img, points, True, (255, 0, 0), 1)
                color_number = classes_ids.index(int(box.cls[0]))
                cv2.fillPoly(link, points, colors[color_number])
        link = blue_road(link)
        link = cv2.cvtColor(link, cv2.COLOR_BGR2GRAY)
        link = (link*(255/np.max(link))).astype(np.uint8)
        h, w = link.shape
        line_row = link[CHECKPOINT, :]
        # cv2.line(link, (0, CHECKPOINT), (w-1, CHECKPOINT), 90, 2)
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
        x1, y1 = center_row, CHECKPOINT
        for pt in link[0, :]: 
            cv2.circle(link, (x1, y1), 5, (179, 179, 179), 3)
        center_of_road = int(math.sqrt(x1*x1+y1*y1))
        print(center_of_road)
        cv2.imshow("Image", link)

segment("/workspace/road/road_1.jpg")
cv2.waitKey(0)
