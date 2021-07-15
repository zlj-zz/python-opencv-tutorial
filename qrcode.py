#!/usr/bin/env python3
import cv2
import numpy as np
from pyzbar.pyzbar import decode


def detect_qrcode(img):
    """Detect qrcode from image.

    Receive a picture and identify all the QR codes or barcodes in it. Mark
    their borders and letters.

    Args:
        img: a image
    """
    for barcode in decode(img):
        print(barcode)
        my_data = barcode.data.decode("utf-8")  # decode for qrcode

        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 255, 0), 5)

        pts2 = barcode.rect
        cv2.putText(
            img,  # image
            my_data,  # text
            (pts2[0], pts2[1]),  # literal direction
            cv2.FONT_HERSHEY_SIMPLEX,  # dot font
            0.9,  # scale
            (255, 255, 0),  # color
            2,  # border
        )


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set the width
    cap.set(4, 480)  # set the height

    while True:
        success, img = cap.read()
        # print(type(img))
        detect_qrcode(img)
        cv2.imshow("res", img)
        cv2.waitKey(1)
