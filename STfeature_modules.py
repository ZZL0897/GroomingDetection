import numpy as np
import cv2

def getColor(Img):
    lower_color = np.array([30, 50, 90])
    upper_color = np.array([200, 200, 200])
    mask = cv2.inRange(Img, lower_color, upper_color)
    mask = 255 - mask
    return mask

def getting_frame_record(frRec, startWin, endWin, fb, newSize, roi, CVNsize, tarRec):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Modified by Augusto Escalante
    # Finally modified by ZL Zhang

    # Subdivide (or not, fb==0) the video frame in order to analyze each arena (for the authors setup that consist on
    # 4 plates each with a fly in the same frame)
    for i in range(startWin, endWin):

        frame = frRec[i, :]
        # Nothing changes from frame to gray, they are identical columns
        gray = frame.reshape(newSize[0] * newSize[1])

        if fb == 0:
            rf = gray
        if fb == 1:
            rf = gray[0:200, 0:200]

        if fb == 2:
            rf = gray[0:200, 200:400]

        if fb == 3:
            rf = gray[200:400, 0:200]

        if fb == 4:
            rf = gray[200:400, 200:400]

        rs = rf

        frameVect = rs.reshape(1, int(newSize[0]) * int(newSize[1]))
        frameVectFloat = frameVect.astype(float)

        if i == startWin:
            previousFrame = frameVectFloat
            frameDiffComm = previousFrame * 0
            frameVectFloatRec = frameVectFloat

        if i > startWin:
            frameDiffComm = frameDiffComm + np.absolute(frameVectFloat - previousFrame)
            frameVectFloatRec = np.vstack((frameVectFloatRec, frameVectFloat))
            previousFrame = frameVectFloat

    # Find the index of the first pixel in the frame that shows the highest difference in intensity respect to the previous frame
    indMaxDiff = np.argmax(frameDiffComm)

    rowMaxDiff = np.floor(indMaxDiff / int(newSize[0]))
    colMaxDiff = indMaxDiff - (rowMaxDiff * int(newSize[0]))

    rowMaxDiff = rowMaxDiff.astype(int)
    colMaxDiff = colMaxDiff.astype(int)

    # Find the value of the pixel in the frame that shows the highest difference in intensity respect to the previous frame
    maxMovement = np.max(frameDiffComm)

    for i in range(0, (endWin - startWin)):

        # Make frameVectFloatRec square
        rs = frameVectFloatRec[i, :].reshape(int(newSize[0]), int(newSize[0]))
        rt = tarRec[i, :].reshape(int(newSize[0]), int(newSize[0]))
        # Calculate a roixroi square around the pixel of maximum intensity difference
        bottomOvershot = 0
        rightOvershot = 0

        topEdge = rowMaxDiff - int(roi * 0.5)
        if topEdge < 0:
            topEdge = 0
        bottomEdge = rowMaxDiff + int(roi * 0.5)
        if bottomEdge > int(newSize[0]):
            bottomOvershot = bottomEdge - int(newSize[0])
            bottomEdge = int(newSize[0])
        leftEdge = colMaxDiff - int(roi * 0.5)
        if leftEdge < 0:
            leftEdge = 0
        rightEdge = colMaxDiff + int(roi * 0.5)
        if rightEdge > int(newSize[0]):
            rightOvershot = rightEdge - int(newSize[0])
            rightEdge = int(newSize[0])

        # Select the roixroi square from the frame
        cfr = rs[topEdge:bottomEdge, leftEdge:rightEdge]
        shapeCfr = cfr.shape
        tfr = rt[topEdge:bottomEdge, leftEdge:rightEdge]

        # Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
        if topEdge == 0:
            rw = np.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = np.vstack((rw, cfr))
            tfr = np.vstack((rw, tfr))
            shapeCfr = cfr.shape
        if bottomOvershot > 0:
            rw = np.zeros((np.absolute(shapeCfr[0] - roi), shapeCfr[1]))
            cfr = np.vstack((cfr, rw))
            tfr = np.vstack((tfr, rw))
            shapeCfr = cfr.shape
        if leftEdge == 0:
            col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = np.hstack((col, cfr))
            tfr = np.hstack((col, tfr))
            shapeCfr = cfr.shape
        if rightOvershot > 0:
            col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1] - roi)))
            cfr = np.hstack((cfr, col))
            tfr = np.hstack((tfr, col))
            shapeCfr = cfr.shape

        # Resize roixroi to CVNsizexCVNsize:
        smallcfr = cv2.resize(cfr, (CVNsize, CVNsize))
        smalltfr = cv2.resize(tfr, (CVNsize, CVNsize))
        cfrVect = smallcfr.reshape(1, CVNsize * CVNsize)
        tfrVect = smalltfr.reshape(1, CVNsize * CVNsize)
        cv2.destroyAllWindows()

        if i == 0:
            cfrVectRec = cfrVect
            tfrVectRec = tfrVect
        if i > 0:
            cfrVectRec = np.vstack((cfrVectRec, cfrVect))
            tfrVectRec = np.vstack((tfrVectRec, tfrVect))

    return maxMovement, cfrVectRec, tfrVectRec, frameVectFloatRec

def  center_of_gravity(cfrVectRec):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Finally modified by ZL Zhang

    sh = np.shape(cfrVectRec)

    F=np.absolute(np.fft.fft(cfrVectRec, axis=0))  #对cfr进行快速傅立叶变换之后取绝对值

    av = np.zeros((1, sh[0])) #建一个行向量，长度为windowST的值
    av[0,:] = np.arange(1, sh[0]+1) #给它赋值，1到ST
    A = np.repeat(av,sh[1], axis=0)  #把上面那个矩阵行数扩展到size的平方，每一行的值都等于上面赋值的内容

    FA = F*np.transpose(A)  #F和A的转置相乘（对应位置的元素直接相乘），F与A的转置维数相同
    # print(FA.shape)
    sF = np.sum(F,axis=0)  #把F每一列的元素加起来
    sFA = np.sum(FA,axis=0)  #把FA每一列的元素加起来
    # print(sF.shape)

    np.seterr(divide='ignore', invalid='ignore') #忽略除法警告
    cG = sFA/sF
    # print(cG.shape)

    return cG  #这里返回的是一个列向量

def create_ST_image(cfrVectRec, tfrVectRec, CVNsize):
    # Quoted from ABRS
    # Written by Primoz Ravbar
    # Modified by Augusto Escalante
    # Finally modified by ZL Zhang

    target = tfrVectRec[1, :]
    target = np.reshape(target, (CVNsize, CVNsize))
    target = target.astype('uint')

    cG = center_of_gravity(cfrVectRec)  # 通过快速傅立叶变换对图片集进行处理

    imRaw = np.reshape(cfrVectRec[1, :], (CVNsize, CVNsize))
    imRaw = imRaw.astype('uint')

    I = np.reshape(cG, (CVNsize, CVNsize)) - 1  # 3月10日修改
    I = np.nan_to_num(I)
    # plt.imshow(I)
    # plt.show()

    IN = np.clip(I, 0, 1)  # 限制图片矩阵元素值在0，1之间

    I_RS = np.reshape(IN, (CVNsize, CVNsize))
    # cv2.imshow('I_RS', I_RS)

    rgbArray = np.zeros((CVNsize, CVNsize, 3), 'uint8')
    rgbArray[..., 0] = I_RS * 255  # blue channel for cv2.imshow()/ red channel for plt.imshow(): Difference with average of windowST frames
    rgbArray[..., 1] = target / 1.8  # green channel for cv2.imshow() and plt.imshow(): Insects informations
    rgbArray[..., 2] = imRaw * 255  # red channel for cv2.imshow()/ blue channel for plt.imshow(): Motionless parts

    imST = rgbArray

    return imST