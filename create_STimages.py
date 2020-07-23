import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tkinter
from tkinter import filedialog
from STfeature_modules import getting_frame_record
from STfeature_modules import create_ST_image
from STfeature_modules import getColor

# This program retrieves all videos in the selected folder and automatically generates STimages of each video

###############################################################
#Parameters set by the user
###############################################################

#Size to which resize the original video (if equal to the longest dimension, 
#no resizing will take place (no resizing will result in slower processing 
#and apparently there is no resolution advantage given the final resizing to 100)):
newSize = [500,500]

#Desired roi size around subject of interest (must be pair) = subarea of the original frame:
roi = 100

#Desired final image size for training the Convolutional Neural Network:
CVNsize = 100

#Set this to any frame in the video:
startFrame = 0
# endFrame = 18920

#Number of frames to calculate the higher scale spatiotemporal feature (red channel):
TimeWindow = 7 # = 0.28 seconds at 25 fps


#fbList = [1,2,3,4]; # works for raw movies with 2x2 arenas (split the frames into 4)
#fbList = [1]; # one arena in the frame #AER: it will still subdivide the arena and take just the upper left square because of function getting_frame_record  
fbList = 0
###############################################################


# show an "Open File" dialog box and returns the path to the selected folder:
root = tkinter.Tk()
root.wm_withdraw()
fileDirPathInput = filedialog.askdirectory() #Choose one folder contains the all videos which need to process
root.destroy()
root.mainloop()

save_path = r'G:\test' #The path to save STimages

video_filename = os.listdir(fileDirPathInput)

for video_name in video_filename:

    print('Processing Video：', video_name)
    video_number = video_name[-7:-4] # Get the video number which is processing

    folder = os.path.exists(save_path + '/v' + video_number) # Create a folder based on the video number
    if not folder:
        os.makedirs(save_path + '/v' + video_number)

    fileDirPathInputName = fileDirPathInput + '/' + video_name # Splicing video file path
    cap = cv2.VideoCapture(fileDirPathInputName)
    endFrame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find out width and height of video:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate amount of padding to add to make frame square:
    if height < width:
        pad = width - height
    if height > width:
        pad = height - width

    # Preallocation:
    prevFrame = np.zeros((newSize[0], newSize[0]))
    frRec = np.zeros((TimeWindow + 1, newSize[0] * newSize[1]))
    tarRec = np.zeros((TimeWindow + 1, newSize[0] * newSize[1]))

    # Read frames one by one from startFrame to endFrame:
    for frameInd in range(startFrame, endFrame, 1):

        cap.set(1, frameInd)
        ret, frame = cap.read()

        # Get the spatial informations
        target_gray = getColor(frame)
        # plt.imshow(target)
        # plt.show()

        # Check frames and convert to grayscale:
        if np.size(np.shape(frame)) >= 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        else:
            print('Corrupt frame with less than 2 dimensions!!')
            gray = np.zeros((width, height))  # Fill the corrupt frame with black

        # Pad frame to make it square adding "pad" black pixels to the bottom or to the right (frame,top,bottom,left,right)
        if height == width:
            gray2 = gray
            target2 = target_gray
        if height < width:
            gray2 = cv2.copyMakeBorder(gray, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # Padding gap with 0
            target2 = cv2.copyMakeBorder(target_gray, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if height > width:
            gray2 = cv2.copyMakeBorder(gray, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            target2 = cv2.copyMakeBorder(target_gray, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Resize frame to newSize if any of the dimensions is different from newSize:
        if newSize[0] != height or newSize[0] != width:
            rs = cv2.resize(gray2, (newSize[0], newSize[1]))
            rst = cv2.resize(target2, (newSize[0], newSize[1]))
        # If one of the dimensions is equal to newSize, no resizing is applied:
        if newSize[0] == height or newSize[0] == width:
            rs = gray2
            rst = target2

        currentFrame = rs.astype(float) / 1
        targetFrame = rst.astype(float) / 1

        frameVect = currentFrame.reshape(1, newSize[0] * newSize[1])  # Tile the current frame into a row vector
        frameVectFloat = frameVect.astype(float)
        tarVect = targetFrame.reshape(1, newSize[0] * newSize[1])
        tarVectFloat = tarVect.astype(float)

        frRecShort = np.delete(frRec, 0, 0)  # Delete the first line of frRec
        frRec = np.vstack((frRecShort, frameVectFloat))  # Adds the row vector of the current frame to the bottom of frRec
        tarRecShort = np.delete(tarRec, 0, 0)
        tarRec = np.vstack((tarRecShort, tarVectFloat))
        # print(tarRec.shape)

        maxMovement, cfrVectRec, tfrVectRec, frameVectFloatRec = getting_frame_record(frRec, 0, TimeWindow, fbList,
                                                                                      newSize, roi, CVNsize,
                                                                                      tarRec)
        imST = create_ST_image(cfrVectRec, tfrVectRec, CVNsize)

        # Using plt to save as RGB, if using cv2, saving as BGR
        plt.imsave(save_path + '/v' + video_number + '/' + str(frameInd) + '.jpg', imST)
        # Saving ST images from first frame to end frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close all opencv stuff
    cap.release()
    cv2.destroyAllWindows()
