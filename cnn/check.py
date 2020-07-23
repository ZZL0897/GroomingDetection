import sys
import threading
import cv2
import pandas as pd
import os
import tkinter
from tkinter import filedialog
from detection_modules import generate_check, copy_mis_motion

# Run this program to manually check and correct

class tt(threading.Thread):  # Create a thread to detect keyboard input
    twice = 2
    input = ''
    input_str = ""

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while 1:
            input_kb = str(sys.stdin.readline()).strip("\n")
            if input_kb:  # stop
                self.twice = 1
                self.input = input_kb
                break
            else:
                self.input_str = input_kb

path = r'D:\code\GroomingDetection'

root = tkinter.Tk()
root.wm_withdraw()
video_dir = filedialog.askopenfilename()
root.destroy()
root.mainloop()
name = video_dir[-7:-4]  # Get the video number
print(name)

cap = cv2.VideoCapture(video_dir)

detection_dir = input('Input the detection file which obtained in detection：')
detection_file = pd.read_csv(path + '/' + detection_dir)

STimage_file = input('Input the video STimage folder：')
STimage_dir = path + '/' + STimage_file
STimages = os.listdir(STimage_dir)
STimages.sort(key=lambda x: int(x[:-4]))  # sort

check_image_dir = r'D:\code\CSB\cnn\TrainingData\check' # The path will save the STimages which are detected error

framerate = 25

# Loading the detection result
start = detection_file['Sframe in video']
end = detection_file['Eframe in video']
motion = detection_file['Behavior']
times = detection_file['Duration']

start_check = []
end_check = []
motion_check = []

for i in range(0, len(start)):

    s = start[i]
    e = end[i]

    print('Behavior：' + motion[i] + '，Duration：' + str(times[i]) + 'sec. Section: ' + str(i+1) + '/' + str(len(start)))
    print('If right input 1, else input the correct behavior type：')

    my_t = tt()
    my_t.start()
    while 1:

        if e - s <= 5 * framerate:

            for frameID in range(s, e):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
                success, frame = cap.read()
                if success:
                    frame = cv2.resize(frame, (960, 560))
                    cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        elif e - s >= 5 * framerate:

            for frameID in range(e - framerate * 8, e):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameID)
                success, frame = cap.read()
                if success:
                    frame = cv2.resize(frame, (960, 560))
                    cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if my_t.twice == 1:
            flag = my_t.input
            break

    if i == 0:
        if flag == '1' or flag == '':
            m = motion[i]
            start_check.append(start[i])
            end_check.append(end[i])
            motion_check.append(motion[i])
        elif flag == 'exit':
            print('Ending check!')
            break
        else:
            m = flag
            start_check.append(start[i])
            end_check.append(end[i])
            motion_check.append(m)

    else:
        if flag == '1' or flag == '':
            m = motion[i]
        elif flag == 'exit':
            print('Ending check!')
            break
        else:
            m = flag

print(start_check)
print(end_check)
print(motion_check)

check = pd.DataFrame(columns=['Smin', 'Ssec', 'Sframe', 'Emin', 'Esec', 'Eframe', 'Duration',
                                      'Continuous frames', 'Behavior', 'Sframe in video', 'Eframe in video'])
check = generate_check(check, start_check, end_check, motion_check, framerate)

check.to_csv(path + '/result/' + name + '_check.csv', index=False)  # Save path and file name

cap.release()
cv2.destroyAllWindows()