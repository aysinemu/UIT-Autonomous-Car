from ultralytics import YOLO
import random
import cv2
import numpy as np
from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket

model = YOLO("/workspace/RoadSeg/weights/best.pt")

def segment(link):
    # if you want all classes
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    conf = 0.9

    results = model.predict(link, conf=conf)
    colors = [255,255,255]
    print(results)
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            # cv2.polylines(img, points, True, (255, 0, 0), 1)
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.fillPoly(link, points, colors[color_number])
    cv2.imshow("Image", link)

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            segment_image = GetSeg()
            segment(raw_image)
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()

