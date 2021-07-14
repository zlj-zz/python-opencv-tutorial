import os
import json
import numpy as np
import cv2 as cv
from pprint import pprint


def id_class_name(class_id, labels_dict):
    """Get class name from `labels`"""
    for key, value in labels_dict.items():
        if class_id == int(key):
            return value


if __name__ == "__main__":
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "frozen_inference_graph.pb")
    pbtxt_path = os.path.join(dir_path, "graph.pbtxt")
    label_path = os.path.join(dir_path, "labels.json")

    # Load class labels.
    labels = json.load(open(label_path))
    pprint(labels)

    # Load model.
    net = cv.dnn.readNetFromTensorflow(model_path, pbtxt_path)

    # Print dnn layer.
    for layer in net.getLayerNames():
        print(layer)
    # exit(0)

    cap = cv.VideoCapture(0)
    while cv.waitKey(1) < 0:
        success, frame = cap.read()
        if frame is None:
            cv.waitKey()
            break

        rows, cols = frame.shape[:-1]

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
                    f"{id_class_name(class_id, labels)}: {round(score * 100, 2)}%",
                    (int(left), int(top) - 10),
                    cv.FONT_HERSHEY_COMPLEX_SMALL,
                    1,
                    (243, 0, 0),
                    2,
                )

        cv.imshow("figure", frame)
