
# ELE510: Image Processing and Computer Vision
# Authors: Mariem El Majdoubi, Aitor Martin, Hugo Philipp
# Project: Removing Fisheye Lens Distortion with Camera Calibration

import os
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#include <opencv2/calib.hpp>
def showChessboardCorners(image, corners):
    # Test: Print image with located corners on top

    for corner in corners:
        cv2.drawMarker(image, (int(corner[0][0]), int(corner[0][1])), color = (255,0,0), markerType=cv2.MARKER_STAR, markerSize = 5, thickness=5)

    plt.imshow(image)
    plt.show()

def calibrateCamera(calibrationImagesPath, patternSize, distortionModel):

    print("\n\n--------------------------------------------------------------------------")
    print("--------------------------- Camera Calibration ---------------------------")
    print("--------------------------------------------------------------------------\n")

    fileList = os.listdir(calibrationImagesPath)

    print(str(len(fileList)) + " files found:")
    print(fileList)

    numImages = int(input("Introduce the number of calibration images to use: "))
    
    # Set object points of the chessboard in the 3D world

    objpoints = np.zeros((1, patternSize[0]*patternSize[1], 3), np.float32)
    objpoints[0,:,:2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

    # Lists for storing points of calibration images

    objectPoints = [] # 3D
    imagePoints = [] # 2D

    i = 0

    for file in fileList[0:numImages]:
        print (file)
        
        i += 1
        calibImage = cv2.imread(calibrationImagesPath + file)
        grayscale = cv2.cvtColor(calibImage, cv2.COLOR_BGR2GRAY)
        imageSize = (calibImage.shape[0], calibImage.shape[1])

        # Find chessboard corners

        if distortionModel == "BrownConrady" :
            retVal, corners = cv2.findChessboardCorners(grayscale, patternSize, cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_ADAPTIVE_THRESH)
        else : 
            retVal, corners = cv2.findChessboardCorners(grayscale, patternSize, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If corners were found correctly, the calibration image is valid
        if retVal:

            print("Chessboard corners sucessfully found on calibration image " + str(i))
            # Append object points
            objectPoints.append(objpoints)
            # Refine the results, to get a more accurate result of the corner locations
            print("Refining corner locations...")
            if distortionModel == "BrownConrady":
                cv2.cornerSubPix(grayscale, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            else : 
                cv2.cornerSubPix(grayscale, corners, (3,3), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            print("Corners refined.")
            # Append image points
            imagePoints.append(corners)
            N_OK = len(objectPoints)

            # Uncomment next line to check chessboard corner detection
            #showChessboardCorners(calibImage, corners)


            

    print("Chessboard corner detection was performed for the calibration images")
    print("Valid images: " + str(len(imagePoints)) + "/" + str(numImages))

    print("Estimating intrinsic and extrinsic camera parameters")

    if distortionModel == "BrownConrady":
        # Perform standard cv2 camera calibration
        ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints, imageSize, None, None)
        print("Intrinsic camera parameters: ")
        fx = cameraMatrix[0][0]
        fy = cameraMatrix[1][1]
        skew = cameraMatrix[0][1]
        xo = cameraMatrix[0][2]
        yo = cameraMatrix[1][2]

        print("fx = " + str(fx))
        print("fy = " + str(fy))
        print("skew = " + str(skew))
        print("xo = " + str(xo))
        print("yo = " + str(yo))

        print("Radial and tangential distortion coefficients: ")
        print("Radial coefficients: ")
        k1 = distCoeffs[0][0]
        k2 = distCoeffs[0][1]
        k3 = distCoeffs[0][4]

        print("k1: " + str(k1))
        print("k2: " + str(k2))
        print("k3: " + str(k3))

        print("Tangential coefficients: ")
        p1 = distCoeffs[0][2]
        p2 = distCoeffs[0][3]

        print("p1: " + str(p1))
        print("p2: " + str(p2))

        #print("Rotation vectors: ")
        #print(rvecs)
        #print("Translation vectors: ")
        #print(tvecs)

        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize)

        print("New Camera Matrix: ")
        print(newCameraMatrix)

        return cameraMatrix, distCoeffs

    elif distortionModel == "Scaramuzza":

        # Perform fisheye camera calibration
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        #retVal, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.fisheye.calibrate(objectPoints, imagePoints, imageSize, None, None)
        cameraMatrix = np.zeros((3, 3))
        distCoeffs = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objectPoints,
                imagePoints,
                grayscale.shape[::-1],
                cameraMatrix,
                distCoeffs,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        

        print("Intrinsic camera parameters: ")

        fx = cameraMatrix[0][0]
        fy = cameraMatrix[1][1]
        skew = cameraMatrix[0][1]
        xo = cameraMatrix[0][2]
        yo = cameraMatrix[1][2]

        print("fx = " + str(fx))
        print("fy = " + str(fy))
        print("skew = " + str(skew))
        print("xo = " + str(xo))
        print("yo = " + str(yo))

        print("Fisheye distortion coefficients: ")

        [theta1, theta2, theta3, theta4] = distCoeffs

        print("theta1: " + str(theta1))
        print("theta2: " + str(theta2))
        print("theta3: " + str(theta3))
        print("theta4: " + str(theta4))

        #print("Rotation vectors: ")
        #print(rvecs)
        #print("Translation vectors: ")
        #print(tvecs)

        return cameraMatrix, distCoeffs


    # Run camera calibration
    # Ask what model the user wants to use: Pinhole (no lens distortion, radial+tangential, fisheye)
    # Print calibration results: model parameters
    # 
    # Save calibration results in a text file. This allows to perform calibration for different cameras.



def undistort(undistortImagesPath,cameraMatrix, distCoeffs, distortionModel):
    
    print("\n\n--------------------------------------------------------------------------")
    print("--------------------------- Undistortion ---------------------------------")
    print("--------------------------------------------------------------------------\n")

    fileList = os.listdir(undistortImagesPath)

    print(str(len(fileList)) + " files found:")
    print(fileList)

    for filename in fileList :
        img = cv2.imread(undistortImagesPath+filename,cv2.IMREAD_COLOR)
        imageSize = (img.shape[1], img.shape[0])

        # Transforms an image to compensate for lens distortion using the camera matrix, 
        # the distortion coefficients and the camera matrix of the distorted image.

        balance = float(input("Input the desired undistortion balance value (choose a value between 0 (automatic crop) and 1 (no crop): "))

        if distortionModel == "BrownConrady":

            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, balance, imageSize)
            map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newCameraMatrix, imageSize, 5)
            res = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)


            # Crop the image to obtained region of interest
            #x, y, w, h = roi
            #res = res[y:y+h, x:x+w]
            cv2.imwrite(undistortImagesPath+"undistorted_"+filename, res)

        elif distortionModel == "Scaramuzza":
            
            new_cameraMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(cameraMatrix, distCoeffs, imageSize, np.eye(3), balance=balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(cameraMatrix, distCoeffs, np.eye(3), new_cameraMatrix, imageSize, cv2.CV_16SC2)
            res = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            cv2.imwrite(undistortImagesPath+"undistorted_"+filename, res)


        
        


print("\n")
print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")
print("-------- Removing Fisheye Lens Distortion with Camera Calibration --------")
print("--------------------------------------------------------------------------")
print("--------------------------------------------------------------------------")

option = -1

while(option != 0):

    print("\n")
    print("##########################################################################")
    print("--------------------------------------------------------------------------")
    print("Select an option:")
    print("1. Calibrate camera")
    print("2. Remove distortion")
    print("0. EXIT")
    print("--------------------------------------------------------------------------")

    option = int(input("Write the desired option: "))

    if option != 0: 
        distortionModel = ""
        model = int(input("Choose distortion model (1. Brown-Conrady or 2. Scaramuzza): "))

        if model == 1:
            distortionModel = "BrownConrady"
        elif model == 2:
            distortionModel = "Scaramuzza"

    if option == 1:

        calibrateCamera("./calibrationImages/", (6,9), distortionModel)

    elif option == 2:

        cameraMatrix, distCoeffs = calibrateCamera("./calibrationImages/", (6,9), distortionModel)
        undistort("./undistortImages/", cameraMatrix, distCoeffs, distortionModel)

    print("\n")
    

print("Exiting...")


