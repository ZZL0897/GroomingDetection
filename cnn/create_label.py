import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm

# Run this program to automatically create label set

path = r'D:\code\GroomingDetection\cnn\TrainingData'

filelist = os.listdir(path + r'\images')

training_set = pd.DataFrame(columns = ['images', 'labels'])

i = 0

for i in tqdm(range(len(filelist))):  #Label the image according to the prefix name and save it

    if re.match('motionless', filelist[i]):  #Motionless
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[0]}))
    elif re.match('head', filelist[i]):  # Head grooming contains feeler grooming, eye grooming and mouth grooming
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[1]}))
    elif re.match('front', filelist[i]):  # Foreleg grooming
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[2]}))
    elif re.match('mid', filelist[i]):  # Grooming contains mid leg
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[3]}))
    elif re.match('hind', filelist[i]):  # Hind leg grooming
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[4]}))
    elif re.match('abdomen', filelist[i]):  # Abdomen grooming
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[5]}))
    elif re.match('wing', filelist[i]):  # Wing grooming
        training_set = training_set.append(pd.DataFrame({'images':[filelist[i]],
                                                         'labels':[6]}))

training_set.to_csv(path + r'\trainging_set.csv', index=False)
print('Label file saved')
# print(filelist)
# print(len(filelist))