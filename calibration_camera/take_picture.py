"""
This is an auxiliary file to help calibrate the camera.
It will call the camera, get the picture and display it in real time.
You can enter any value in the console to save a picture.
"""

import cv2
import threading


if_save = False
cap_num = int(input("Input the camare number:"))
name = int(input("Input start name, use number:"))

cap = cv2.VideoCapture(cap_num)


def save():
    global if_save
    while True:
        input("Input any to save a image:")
        if_save = True


threading.Thread(target=save, daemon=True).start()


while cv2.waitKey(1) != ord("q"):
    _, frame = cap.read()
    if if_save:
        img_name = str(name) + ".jpg"
        cv2.imwrite(img_name, frame)
        print("Save {} successful.".format(img_name))
        name += 1
        if_save = False
    cv2.imshow("", frame)
