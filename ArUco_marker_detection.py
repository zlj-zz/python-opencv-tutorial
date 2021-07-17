import cv2 as cv
import numpy as np


if __name__ == "__main__":
    cap_num = 0
    cap = cv.VideoCapture(cap_num)

    # Get ArUco marker dict that can be detected.
    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    # Get ArUco marker params.
    aruco_params = cv.aruco.DetectorParameters_create()

    calibrationParams = cv.FileStorage("calibrationFileName.xml", cv.FILE_STORAGE_READ)
    # Get distance coefficient.
    dist_coeffs = calibrationParams.getNode("distCoeffs").mat()
    print(dist_coeffs)
    height = cap.get(4)
    focal_length = width = cap.get(3)
    center = [width / 2, height / 2]
    # Calculate the camera matrix.
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    print(camera_matrix)

    while cv.waitKey(1) & 0xFF != ord("q"):
        success, img = cap.read()
        if not success:
            print("It seems that the image cannot be acquired correctly.")
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Detect ArUco marker.
        corners, ids, rejectImaPoint = cv.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if len(corners) > 0:
            if ids is not None:
                # print('corners:', corners, 'ids:', ids)
                print("ids:", ids)
                ret = cv.aruco.estimatePoseSingleMarkers(
                    corners, 0.05, camera_matrix, dist_coeffs
                )
                # print(ret)
                (rvec, tvec) = (ret[0], ret[1])
                (rvec - tvec).any()

                print("rvec:", rvec, "tvec:", tvec)

                for i in range(rvec.shape[0]):
                    cv.aruco.drawDetectedMarkers(img, corners)
                    cv.aruco.drawAxis(
                        img,
                        camera_matrix,
                        dist_coeffs,
                        rvec[i, :, :],
                        tvec[i, :, :],
                        0.03,
                    )

        cv.imshow("Image", img)
