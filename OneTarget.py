import numpy as np
import cv2

url = "http://10.4.105.218:8080/video"
cap = cv2.VideoCapture(url)

imgTarget = cv2.imread('img/cat.png')
imgTarget = cv2.resize(imgTarget, (500,500))
myVid = cv2.VideoCapture('gif/cat.gif')
detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (hT, wT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)


while True:
    success, imgWebcam = cap.read()
    imgAug = imgWebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    if detection == False:
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0
    else:
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success, imgVideo = myVid.read()
        imgVideo = cv2.resize(imgVideo, (wT, hT))



    bf = cv2.BFMatcher()
    macthes = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in macthes:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags = 2)

    if len(good) > 20 :
        detection = True
        srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)
        print('Render success')

        pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)

        imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew, [np.int32(dst)], [255, 255, 255])
        maskInv = cv2.bitwise_not(maskNew)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv)
        imgAug = cv2.bitwise_or(imgWarp, imgAug)

    cv2.imshow('imgAug1', imgAug)
    cv2.imshow('imgWarp', imgWarp)

    # cv2.imshow('img2', img2)
    # cv2.imshow('imgFeatures1', imgFeatures)
    # cv2.imshow('ImgTarget', imgTarget)
    # cv2.imshow('myVid', imgVideo)


    cv2.imshow('Webcam1', imgWebcam)
    cv2.waitKey(1)
    frameCounter += 1