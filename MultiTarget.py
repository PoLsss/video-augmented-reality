import numpy as np
import cv2

url = "http://10.4.105.218:8080/video"
cap = cv2.VideoCapture(url)

imgTarget = cv2.imread('img/cat.png')
imgTarget1 = cv2.imread('img/golila.png')
imgTarget2 = cv2.imread('img/anh_shiba.png')
imgTarget3 = cv2.imread('img/tik.png')

# imgTarget = cv2.resize(imgTarget, (500,500))
# imgTarget1 = cv2.resize(imgTarget1, (500,500))
# imgTarget2 = cv2.resize(imgTarget2, (500,500))
# imgTarget3 = cv2.resize(imgTarget3, (500,500))

myVid = cv2.VideoCapture('gif/cat.gif')
myVid1 = cv2.VideoCapture('gif/kingkong1.mp4')
myVid2 = cv2.VideoCapture('gif/gif-Shiba-nhay-1.gif')
myVid3 = cv2.VideoCapture('gif/tik.mp4')


detection = False
frameCounter = 0

detection1 = False
frameCounter1 = 0

detection2 = False
frameCounter2 = 0

detection3 = False
frameCounter3 = 0

success, imgVideo = myVid.read()
success1, imgVideo1 = myVid1.read()
success2, imgVideo2 = myVid2.read()
success3, imgVideo3 = myVid3.read()


hT, wT, cT = imgTarget.shape
hT1, wT1, cT1 = imgTarget1.shape
hT2, wT2, cT2 = imgTarget2.shape
hT3, wT3, cT3 = imgTarget3.shape
imgVideo = cv2.resize(imgVideo, (hT, wT))
imgVideo1 = cv2.resize(imgVideo1, (hT1, wT1))
imgVideo2 = cv2.resize(imgVideo2, (hT2, wT2))
imgVideo3 = cv2.resize(imgVideo3, (hT3, wT3))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
kp_1, des_1 = orb.detectAndCompute(imgTarget1, None)
kp_2, des_2 = orb.detectAndCompute(imgTarget2, None)
kp_3, des_3 = orb.detectAndCompute(imgTarget3, None)
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

    if detection1 == False:
        myVid1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter1 = 0
    else:
        if frameCounter1 == myVid1.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter1 = 0
        success, imgVideo1 = myVid1.read()
        imgVideo1 = cv2.resize(imgVideo1, (wT1, hT1))

    if detection2 == False:
        myVid2.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter2 = 0
    else:
        if frameCounter2 == myVid2.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter2 = 0
        success, imgVideo2 = myVid2.read()
        imgVideo2 = cv2.resize(imgVideo2, (wT2, hT2))

    if detection3 == False:
        myVid3.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter3 = 0
    else:
        if frameCounter3 == myVid3.get(cv2.CAP_PROP_FRAME_COUNT):
            myVid3.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter3 = 0
        success, imgVideo3 = myVid3.read()
        imgVideo3 = cv2.resize(imgVideo3, (wT3, hT3))

    bf = cv2.BFMatcher()
    macthes = bf.knnMatch(des1, des2, k=2)
    macthes1 = bf.knnMatch(des_1, des2, k=2)
    macthes2 = bf.knnMatch(des_2, des2, k=2)
    macthes3 = bf.knnMatch(des_3, des2, k=2)
    good = []
    good1 = []
    good2 = []
    good3 = []
    for m, n in macthes:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))

    for m, n in macthes1:
        if m.distance < 0.75 * n.distance:
            good1.append(m)
    print(len(good1))
    for m, n in macthes2:
        if m.distance < 0.75 * n.distance:
            good2.append(m)
    print(len(good2))
    for m, n in macthes3:
        if m.distance < 0.75 * n.distance:
            good3.append(m)
    print(len(good3))

    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags = 2)
    imgFeatures1 = cv2.drawMatches(imgTarget1, kp_1, imgWebcam, kp2, good1, None, flags = 2)
    imgFeatures2 = cv2.drawMatches(imgTarget2, kp_2, imgWebcam, kp2, good2, None, flags=2)
    imgFeatures3 = cv2.drawMatches(imgTarget3, kp_3, imgWebcam, kp2, good3, None, flags=2)

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

    if len(good1) > 20 :
        detection1 = True
        srcPts1 = np.float32([kp_1[m.queryIdx].pt for m in good1]).reshape(-1, 1, 2)
        dstPts1 = np.float32([kp2[m.trainIdx].pt for m in good1]).reshape(-1, 1, 2)
        matrix1, mask1 = cv2.findHomography(srcPts1, dstPts1, cv2.RANSAC, 5)
        print('Render success')

        pts1 = np.float32([[0, 0], [0, hT1], [wT1, hT1], [wT1, 0]]).reshape(-1, 1, 2)
        dst1 = cv2.perspectiveTransform(pts1, matrix1)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst1)], True, (255, 0, 255), 3)

        imgWarp1 = cv2.warpPerspective(imgVideo1, matrix1, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew1 = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew1, [np.int32(dst1)], [255, 255, 255])
        maskInv1 = cv2.bitwise_not(maskNew1)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv1)
        imgAug = cv2.bitwise_or(imgWarp1, imgAug)

    if len(good2) > 20 :
        detection2 = True
        srcPts2 = np.float32([kp_2[m.queryIdx].pt for m in good2]).reshape(-1, 1, 2)
        dstPts2 = np.float32([kp2[m.trainIdx].pt for m in good2]).reshape(-1, 1, 2)
        matrix2, mask2 = cv2.findHomography(srcPts2, dstPts2, cv2.RANSAC, 5)
        print('Render success')

        pts2 = np.float32([[0, 0], [0, hT2], [wT2, hT2], [wT2, 0]]).reshape(-1, 1, 2)
        dst2 = cv2.perspectiveTransform(pts2, matrix2)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst2)], True, (255, 0, 255), 3)

        imgWarp2 = cv2.warpPerspective(imgVideo2, matrix2, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew2 = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew2, [np.int32(dst2)], [255, 255, 255])
        maskInv2 = cv2.bitwise_not(maskNew2)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv2)
        imgAug = cv2.bitwise_or(imgWarp2, imgAug)

    if len(good3) > 20 :
        detection3 = True
        srcPts3 = np.float32([kp_3[m.queryIdx].pt for m in good3]).reshape(-1, 1, 2)
        dstPts3 = np.float32([kp2[m.trainIdx].pt for m in good3]).reshape(-1, 1, 2)
        matrix3, mask3 = cv2.findHomography(srcPts3, dstPts3, cv2.RANSAC, 5)
        print('Render success')

        pts3 = np.float32([[0, 0], [0, hT3], [wT3, hT3], [wT3, 0]]).reshape(-1, 1, 2)
        dst3 = cv2.perspectiveTransform(pts3, matrix3)
        img2 = cv2.polylines(imgWebcam, [np.int32(dst3)], True, (255, 0, 255), 3)

        imgWarp3 = cv2.warpPerspective(imgVideo3, matrix3, (imgWebcam.shape[1], imgWebcam.shape[0]))

        maskNew3 = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
        cv2.fillPoly(maskNew3, [np.int32(dst3)], [255, 255, 255])
        maskInv3 = cv2.bitwise_not(maskNew3)
        imgAug = cv2.bitwise_and(imgAug, imgAug, mask = maskInv3)
        imgAug = cv2.bitwise_or(imgWarp3, imgAug)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('imgAug', imgAug)
    # cv2.imshow('imgWarp', imgWarp)
    # cv2.imshow('img2', img2)
    # cv2.imshow('imgFeatures1', imgFeatures)
    # cv2.imshow('imgFeatures2', imgFeatures1)
    # cv2.imshow('ImgTarget', imgTarget)
    # cv2.imshow('myVid', imgVideo)
    cv2.imshow('Webcam1', imgWebcam)
    cv2.waitKey(1)
    frameCounter += 1
    frameCounter1 += 1
    frameCounter2 += 1
    frameCounter3 += 1