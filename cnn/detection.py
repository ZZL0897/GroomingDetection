import detection_modules
import numpy as np
from keras.models import load_model
import os
import pandas as pd

# This program retrieves all STimages folders in the path and automatically generates detection results

model_dir = r'D:\code\GroomingDetection\cnn\TrainingData\test'
model = load_model(model_dir)
framerate = 25 # Input your framerate

path = r'H:\ABRS\3.25'  # The folders path which saves several videos' STimages

file_name = os.listdir(path)

for file_id in file_name:

    img_dir = path + '/' + file_id

    predict = detection_modules.predict_images(img_dir, model)
    predict = np.array(predict)

    i = 0
    flag = 0
    start = []
    end = []
    motion = []
    for i in range(0, len(predict)):
        if i == 0:
            now = predict[i]
            start.append(i)
            motion.append(predict[i])
        if i == len(predict) - 1:
            end.append(i)
        elif now == predict[i]:
            flag = 0
        elif now != predict[i]:
            if flag < 10:
                flag += 1
            elif flag >= 10:
                flag = 0
                now = predict[i]
                motion.append(predict[i])
                end.append(i - 10)
                start.append(i - 10)

    detection = pd.DataFrame(columns=['Smin', 'Ssec', 'Sframe', 'Emin', 'Esec', 'Eframe', 'Duration',
                                      'Continuous frames', 'Behavior', 'Sframe in video', 'Eframe in video'])  # Saving result
    detection = detection_modules.generate_csv(detection, start, end, motion, framerate)

    detection.to_csv(r'D:\code\GroomingDetection\cnn\TrainingData' + '/detection_' + file_id + '.csv', index=False) # Saving as .csv

    print(start)
    print(end)
    print(motion)
    print('Total' + str(len(motion)) + 'behaviors')