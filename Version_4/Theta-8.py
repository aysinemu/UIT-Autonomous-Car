from client_lib import GetStatus,GetRaw,GetSeg,AVControl,CloseSocket
import cv2

count = 0

if __name__ == "__main__":
    try:
        while True:
            state = GetStatus()
            raw_image = GetRaw()
            # segment_image = GetSeg()

            # print(state)
            # cv2.imshow('raw_image', raw_image)
            # cv2.imshow('segment_image', segment_image)

            # maxspeed = 90, max steering angle = 25
            # AVControl(speed=-10, angle=-10)
            image_name = "./img/img_{}.jpg".format(count)
            count += 1
            cv2.imwrite(image_name, raw_image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        CloseSocket()
