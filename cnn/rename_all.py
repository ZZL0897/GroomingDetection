import os
import shutil

# Run this program to automatically rename all STimages in every folder and copy them to the images folder

def rename(path, name):
    filelist = os.listdir(path)
    total_num = len(filelist)
    i = 0
    for item in filelist:
        if item.endswith('.jpg'):  # The suffix name of the image file that needs to be changed
            src = os.path.join(os.path.abspath(path), item)
            dst = os.path.join(os.path.abspath(path), str(name) + '-' + str(i) + '.jpg')  # Changed suffix
            try:
                os.rename(src, dst)
                print('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue
    print('total %d to rename & converted %d jpgs' % (total_num, i))

def copy(path, copy_path):
    filelist = os.listdir(path)
    for file in filelist:
        shutil.copy(path + '/' + file, copy_path)

abdomen = r'D:\code\GroomingDetection\cnn\TrainingData\abdomen'
front = r'D:\code\GroomingDetection\cnn\TrainingData\front'
head = r'D:\code\GroomingDetection\cnn\TrainingData\head'
hind = r'D:\code\GroomingDetection\cnn\TrainingData\hind'
mid = r'D:\code\GroomingDetection\cnn\TrainingData\mid'
motionless = r'D:\code\GroomingDetection\cnn\TrainingData\motionless'
wing = r'D:\code\GroomingDetection\cnn\TrainingData\wing'
copy_path = r'D:\code\GroomingDetection\cnn\TrainingData\images'

rename(abdomen, 'abdomen')
rename(front, 'front')
rename(head, 'head')
rename(hind, 'hind')
rename(mid, 'mid')
rename(motionless, 'motionless')
rename(wing, 'wing')

copy(abdomen, copy_path)
copy(front, copy_path)
copy(head, copy_path)
copy(hind, copy_path)
copy(mid, copy_path)
copy(motionless, copy_path)
copy(wing, copy_path)