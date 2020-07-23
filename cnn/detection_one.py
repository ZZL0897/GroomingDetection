import detection_modules
import numpy as np
from keras.models import load_model
import os
import pandas as pd

# This program automatically generates detection result according to the path

model_dir = r'D:\code\GroomingDetection\cnn\TrainingData\model'
model = load_model(model_dir)
framerate = 25 # Input your framerate

path = r'D:\video\110'  # # The folders path which saves one video's STimages
video_id = path[-3::]

img_dir = os.listdir(path)

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

detection.to_csv(r'D:\code\GroomingDetection\cnn\TrainingData' + '/detection_' + video_id + '.csv', index=False) # Saving as .csv

print(start)
print(end)
print(motion)
print('Total' + str(len(motion)) + 'behaviors')