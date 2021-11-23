import os
import json
import random
import sys
import glob
from random import shuffle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--data_path", type=str, default=None)
args = parser.parse_args()
path_list = []

def msu_process(data_dir, label_save_dir):
    test_list = []
    # data_label for msu
    for line in open(data_dir + 'test_sub_list.txt', 'r'):
        test_list.append(line[0:2])
    train_list = []
    for line in open(data_dir + 'train_sub_list.txt', 'r'):
        train_list.append(line[0:2])
    print(test_list)
    print(train_list)
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    print_final_json = []
    video_final_json = []
    # label_save_dir = './labels/msu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_print = open(label_save_dir + 'print_label.json', 'w')
    f_video = open(label_save_dir + 'video_label.json', 'w')
    dataset_path = data_dir
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if(flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        video_num = path_list[i].split('/')[-2].split('_')[1][-2:]
        if (video_num in train_list):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if(label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
        if path_list[i].split('/')[-2].find('video') != -1:
            video_final_json.append(dict)
        elif path_list[i].split('/')[-2].find('printed') != -1:
            print_final_json.append(dict)
        elif label == 1:
            rand = random.random()
            if rand > 0.5:
                video_final_json.append(dict)
            else:
                print_final_json.append(dict)
    print('\nMSU: ', len(path_list))
    print('MSU(train): ', len(train_final_json))
    print('MSU(test): ', len(test_final_json))
    print('MSU(all): ', len(all_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    print('MSU(video): ', len(video_final_json))
    print('MSU(print): ', len(print_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(video_final_json, f_video, indent=4)
    f_video.close()
    json.dump(print_final_json, f_print, indent=4)
    f_print.close()
def casia_process(data_dir, label_save_dir):
    train_final_json = []
    test_final_json = []
    dev_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    print_final_json = []
    video_final_json = []
    other_final_json = []
    # label_save_dir = './labels/casia/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_dev = open(label_save_dir + 'dev_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_print = open(label_save_dir + 'print_label.json', 'w')
    f_video = open(label_save_dir + 'video_label.json', 'w')
    dataset_path = data_dir
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-2]
        if (flag == '1' or flag == '2' or flag == 'HR_1'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if path_list[i].find('/train_release/')!=-1:
            train_final_json.append(dict)
        elif path_list[i].find('/test_release/')!=-1:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
        flag = path_list[i].split('/')[-2]
        if label == 0:
            if flag == '3' or flag == '4' or flag == '5' or flag == '6' or flag == 'HR_2' or flag == 'HR_3':
                print_final_json.append(dict)
            elif flag == '7' or flag == '8' or flag == 'HR_4':
                video_final_json.append(dict)
            else:
                # print(other_final_json)
                other_final_json.append(dict)
        else:
            prob = random.random()
            if prob>=0.5:
                print_final_json.append(dict)
            else:
                video_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(dev): ', len(dev_final_json))
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    print('Casia(print): ', len(print_final_json))
    print('Casia(video): ', len(video_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(dev_final_json, f_dev, indent=4)
    f_dev.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(print_final_json, f_print, indent=4)
    f_print.close()
    json.dump(video_final_json, f_video, indent=4)
    f_video.close()
def idiap_process(data_dir, label_save_dir):
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    print_final_json = []
    video_final_json = []
    # label_save_dir = './labels/idiap/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_print = open(label_save_dir + 'print_label.json', 'w')
    f_video = open(label_save_dir + 'video_label.json', 'w')
    dataset_path = data_dir
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if (flag != -1):
            label = 1
        elif path_list[i].find('/enroll/') != -1:
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if (path_list[i].find('/train/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/devel/') != -1):
            valid_final_json.append(dict)
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)

        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
        if label == 0:
            if path_list[i].split('/')[-2].split('_')[1] == 'highdef' or path_list[i].split('/')[-2].split('_')[1] == 'mobile':
                video_final_json.append(dict)
            elif path_list[i].split('/')[-2].split('_')[1] == 'print':
                print_final_json.append(dict)
            else:
                print(path_list[i])
        else:
            rand = random.random()
            if rand >= 0.5:
                video_final_json.append(dict)
            else:
                print_final_json.append(dict)
    print('\nReplay: ', len(path_list))
    print('Replay(train): ', len(train_final_json))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(test): ', len(test_final_json))
    print('Replay(all): ', len(all_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    print('Replay(video): ', len(video_final_json))
    print('Replay(print): ', len(print_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(video_final_json, f_video, indent=4)
    f_video.close()
    json.dump(print_final_json, f_print, indent=4)
    f_print.close()
def oulu_process(data_dir, label_save_dir):
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    print_final_json = []
    video_final_json = []
    # label_save_dir = './labels/oulu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    f_print = open(label_save_dir + 'print_label.json', 'w')
    f_video = open(label_save_dir + 'video_label.json', 'w')
    dataset_path = data_dir
    path_list = glob.glob(dataset_path + '**/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = int(path_list[i].split('/')[-2].split('_')[-1])
        if (flag == 1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        all_final_json.append(dict)
        if (path_list[i].find('/Train_files/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/Dev_files/') != -1):
            valid_final_json.append(dict)
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)

        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
        if flag == 2 or flag == 3:
            print_final_json.append(dict)
        elif flag == 4 or flag == 5:
            video_final_json.append(dict)
        elif flag == 1:
            prob = random.random()
            if prob >= 0.5:
                print_final_json.append(dict)
            else:
                video_final_json.append(dict)
    print('\nOulu: ', len(path_list))
    print('Oulu(train): ', len(train_final_json))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(test): ', len(test_final_json))
    print('Oulu(all): ', len(all_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    print('Oulu(print): ', len(print_final_json))
    print('Oulu(video): ', len(video_final_json))
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()
    json.dump(print_final_json, f_print, indent=4)
    f_print.close()
    json.dump(video_final_json, f_video, indent=4)
    f_video.close()



if __name__=="__main__":
    """
    change your data path
    path_list: [msu_path, casia_path, idiap_path, oulu_path]
    """
    path_list = args.data_path
    msu_process(path_list[0], './labels/msu/')
    casia_process(path_list[1], './labels/casia/')
    idiap_process(path_list[2], './labels/idiap/')
    oulu_process(path_list[3], './labels/oulu/')



    