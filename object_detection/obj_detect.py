"""
Detect object.
env:
    python2/python3
    opencv4.*
"""

import os
import json
import time
import numpy as np
import cv2 as cv
from pprint import pprint


def load_labels(path):
    """Load class labels from json file."""
    labels = json.load(open(path))
    return labels


def id_class_name(class_id, labels_dict):
    """Get class name from `labels`"""
    for key, value in labels_dict.items():
        if class_id == int(key):
            return value


def draw_size_and_fps(img, size, fps):
    cv.putText(
        img,
        "Size: {size}".format(size=size),
        (0, 20),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (0, 0, 240),
        1,
    )
    cv.putText(
        img,
        "FPS: {fps}".format(fps=fps),
        (0, 40),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (0, 0, 240),
        1,
    )


if __name__ == "__main__":
    # Get needed pathes.
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "frozen_inference_graph.pb")
    pbtxt_path = os.path.join(dir_path, "graph.pbtxt")
    label_path = os.path.join(dir_path, "labels.json")

    # Load class labels.
    labels = load_labels(label_path)
    pprint(labels)

    # Load model.
    net = cv.dnn.readNetFromTensorflow(model_path, pbtxt_path)

    # Print dnn layer.
    for layer in net.getLayerNames():
        print(layer)
    # exit(0)

    # Init and open camera.
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        cap.open()
    cap_size = (cap.get(3), cap.get(4))

    while cv.waitKey(1) < 0:
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            cv.waitKey()
            break

        rows, cols = frame.shape[:-1]

        # Resize image and swap BGR to RGB.
        blob = cv.dnn.blobFromImage(
            frame,
            size=(300, 300),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        # print("First Blob: {}".format(blob.shape))

        # Detecting.
        net.setInput(blob)
        out = net.forward()

        # Processing result.
        for detection in out[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.6:
                # print(detection)
                class_id = detection[1]
                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                cv.rectangle(
                    frame,
                    (int(left), int(top)),
                    (int(right), int(bottom)),
                    (0, 230, 0),
                    thickness=2,
                )
                cv.putText(
                    frame,
                    "{class_name}: {percent}%".format(
                        class_name=id_class_name(class_id, labels),
                        percent=round(score * 100, 2),
                    ),
                    (int(left), int(top) - 10),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (243, 0, 0),
                    2,
                )

        # Draw capture size and FPS.
        draw_size_and_fps(frame, cap_size, round(1.0 / (time.time() - start_time), 2))

        # Show.
        cv.imshow("figure", frame)
