from keras.preprocessing import image
import numpy as np
import os
import pandas as pd
import cv2
import re
import shutil

def load_data(path):
    data = pd.read_csv(path + r'\trainging_set.csv')  # Read csv

    image_data = np.empty((len(data['images']), 100, 100, 3))  # Initialize image storage matrix
    label_data = []  # Initializing tag storage list
    i = 0

    for file in data['images']:  # Read the images according to the image name stored in the CSV file
        # print(file)
        img = cv2.imread(path + "/images/" + file)[:, :, ::-1]  # Read as RGB, if not ,read as BGR
        img = cv2.resize(img, (100, 100))
        # print(img)
        # plt.imshow(img)
        # plt.show()
        image_data[i] = img
        # print(image_data[i])
        # plt.imshow(image_data[i])
        # plt.show()
        i = i + 1

    for label in data['labels']:  # Save the label information to the list, and image_data subscript correspondence
        label_data.append(int(label))

    label_data = np.array(label_data)  # Convert to numpy array

    return image_data, label_data

def load_data_vgg(path):
    data = pd.read_csv(path + r'\trainging_set.csv')

    image_data = np.empty((len(data['images']), 224, 224, 3))
    label_data = []
    i = 0

    for file in data['images']:
        # print(file)
        img = cv2.imread(path + "/images/" + file)[:, :, ::-1]
        img = cv2.resize(img, (224, 224))
        # print(img)
        # plt.imshow(img)
        # plt.show()
        image_data[i] = img
        # print(image_data[i])
        # plt.imshow(image_data[i])
        # plt.show()
        i = i + 1

    for label in data['labels']:
        label_data.append(int(label))

    label_data = np.array(label_data)

    return image_data, label_data

# Detection folder
def predict_images(img_dir, model):
    img = []
    files = os.listdir(img_dir)
    files.sort(key=lambda x:int(x[:-4]))  # sort
    for f in files:
        image_path = os.path.join(img_dir, f)
        if os.path.isfile(image_path):
            images = image.load_img(image_path)
            x = image.img_to_array(images)
            x = cv2.resize(x, (100, 100))
            x = np.expand_dims(x, axis=0)
            img.append(x)
    x = np.concatenate([x for x in img])

    y = model.predict(x)  # The prediction result is a one-hot matrix
    predict = [np.argmax(one_hot) for one_hot in y]  # Converting one-hot matrix to a list
    return predict

# Detection one image
def predict_one_image(img_path, model):
    img = image.load_img(img_path, target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds

# Converts the number of frames to time and returns three integers: minutes, seconds, and milliseconds
def frames_to_timecode(framerate,frames):

    time =  '{0:02d}-{1:02d}-{2:02d}'.format(int(frames / (60 * framerate) % 60),
                                            int(frames / framerate % 60),
                                            int((frames % framerate) * (1000/framerate)))
    time = re.split('-', time)
    time = list(map(int, time))
    if time[0] == 0:
        del time[0]
        length = 2
        return length, time
    else:
        length = 3
        return length, time

# Converts the number of frames to time and returns three integers: minutes, seconds and frames
def frames_to_timecode2(framerate,frames):

    time =  '{0:02d}-{1:02d}-{2:02d}'.format(int(frames / (60 * framerate) % 60),
                                            int(frames / framerate % 60),
                                            int(frames % framerate))
    time = re.split('-', time)
    time = list(map(int, time))
    min = time[0]
    sec = time[1]
    divide_frame = time[2]

    return min, sec, divide_frame

def get_frameID(time, framerate):  # Get frames by time
    time = re.split('-', time)
    time = list(map(int, time))
    frameID = time[0]*60*framerate + time[1]*framerate + time[2]/(1000/framerate)
    frameID = int(frameID)
    return frameID

def generate_csv(detection, start_frame, end_frame, motion, framerate):
    motion_name = []
    for i in motion:
        if i == 0:
            motion_name.append('0')
        elif i == 1:
            motion_name.append('head')
        elif i == 2:
            motion_name.append('front')
        elif i == 3:
            motion_name.append('mind')
        elif i == 4:
            motion_name.append('hind')
        elif i == 5:
            motion_name.append('abdomen')
        elif i == 6:
            motion_name.append('wing')

    for i in range(0, len(start_frame)):
        smin, ssec, sdivide_frame = frames_to_timecode2(framerate, start_frame[i])
        emin, esec, edivide_frame = frames_to_timecode2(framerate, end_frame[i])
        detection = detection.append(pd.DataFrame({'Smin': [smin],
                                                   'Ssec': [ssec],
                                                   'Sframe': [sdivide_frame],
                                                   'Emin': [emin],
                                                   'Esec': [esec],
                                                   'Eframe': [edivide_frame],
                                                   'Duration': [(end_frame[i] - start_frame[i])/25],
                                                   'Continuous frames': [end_frame[i] - start_frame[i]],
                                                   'Behavior': [motion_name[i]],
                                                   'Sframe in video':[start_frame[i]],
                                                   'Eframe in video':[end_frame[i]]}))
    return detection

def generate_check(check, start_check, end_check, motion_check, framerate):
    for i in range(0, len(start_check)):
        smin, ssec, sdivide_frame = frames_to_timecode2(framerate, start_check[i])
        emin, esec, edivide_frame = frames_to_timecode2(framerate, end_check[i])
        check = check.append(pd.DataFrame({'Smin': [smin],
                                           'Ssec': [ssec],
                                           'Sframe': [sdivide_frame],
                                           'Emin': [emin],
                                           'Esec': [esec],
                                           'Eframe': [edivide_frame],
                                           'Duration': [(end_check[i] - start_check[i])/25],
                                           'Continuous frames': [end_check[i] - start_check[i]],
                                           'Behavior': [motion_check[i]],
                                           'Sframe in video':[start_check[i]],
                                           'Eframe in video':[end_check[i]]}))
    return check

def copy(path, filelist, index, copy_path):
    for i in index:
        shutil.copy(path + '/' + filelist[i], copy_path)

def copy_mis_motion(start, end, motion_name, image_dir, image_list, save_dir):
    avg = int((start + end)/2)
    copy_index = range(avg-2, avg+3)
    if re.search('mid', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/mid')
    elif re.search('eye', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/head')
    elif re.search('feeler', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/head')
    elif re.search('mouth', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/head')
    elif re.search('front', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/front')
    elif re.search('abdomen', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/abdomen')
    elif re.search('ovipositor', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/abdomen')
    elif re.search('hind', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/hind')
    elif re.search('wing', motion_name):
        copy(image_dir, image_list, copy_index, save_dir + '/wing')
