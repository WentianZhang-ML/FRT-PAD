import json
import math
from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
from torchvision import transforms as T
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
class TODDataset(Dataset):
    def __init__(self, data, transforms=None, train=True):
        self.data = data
        self.train = train
        if transforms is None:
            if self.train:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor()
                ])
            else:
                self.transforms = T.Compose([
                    T.ToTensor()
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]['photo_path']
        img_label = self.data[item]['photo_label']
        img = Image.open(img_path)
        videoID = self.data[item]['photo_belong_to_video_ID']
        img = self.transforms(img)
        return img, img_label, videoID

def load_data(root_path, flag, dataset_name):
    if(flag == 0): # select the training images
        label_path = root_path + '/' + dataset_name + '/train_label.json'
    elif(flag == 1): # select the testing images
        label_path = root_path + '/' + dataset_name + '/test_label.json'
    elif (flag==2): # select all the images
        label_path = root_path + '/' + dataset_name + '/all_label.json'
    all_label_json = json.load(open(label_path, 'r'))
    return all_label_json
def sample_frames(all_label_json, num_frames):
    length = len(all_label_json)
    saved_frame_prefix = '/'.join(all_label_json[0]['photo_path'].split('/')[:-1])
    final_json = []
    video_number = 0
    single_video_frame_list = []
    single_video_frame_num = 0
    single_video_label = 0
    for i in range(length):
        photo_path = all_label_json[i]['photo_path']
        photo_label = all_label_json[i]['photo_label']
        frame_prefix = '/'.join(photo_path.split('/')[:-1])
        # the last frame
        if (i == length - 1):
            photo_frame = int(photo_path.split('/')[-1].split('.')[0])
            single_video_frame_list.append(photo_frame)
            single_video_frame_num += 1
            single_video_label = photo_label
        # a new video, so process the saved one
        if (frame_prefix != saved_frame_prefix or i == length - 1):
            # [1, 2, 3, 4,.....]
            single_video_frame_list.sort()
            frame_interval = math.floor(single_video_frame_num / num_frames)
            for j in range(num_frames):
                dict = {}
                try:
                    dict['photo_path'] = saved_frame_prefix + '/' + str(
                        single_video_frame_list[6 + j * frame_interval]) + '.jpg'
                except Exception:
                    dict['photo_path'] = saved_frame_prefix + '/' + str(
                        single_video_frame_list[0 + j * frame_interval]) + '.jpg'
                dict['photo_label'] = single_video_label
                dict['photo_belong_to_video_ID'] = video_number
                final_json.append(dict)
            video_number += 1
            saved_frame_prefix = frame_prefix
            single_video_frame_list.clear()
            single_video_frame_num = 0
        # get every frame information
        photo_frame = int(photo_path.split('/')[-1].split('.')[0])
        single_video_frame_list.append(photo_frame)
        single_video_frame_num += 1
        single_video_label = photo_label
    return final_json
def get_dataset(root_path, train_data, train_num_frames, test_data, test_num_frames, batchsize):
    data1 = load_data(root_path, flag=2, dataset_name='oulu')
    data2 = load_data(root_path, flag=2, dataset_name='casia')
    data3 = load_data(root_path, flag=2, dataset_name='idiap')
    data4 = load_data(root_path, flag=2, dataset_name='msu')

    if train_data == 'OULU_MSU':
        train_data_all = sample_frames(data1+data4, num_frames=train_num_frames)
    elif train_data == 'CASIA_Idiap':
        train_data_all = sample_frames(data2 + data3, num_frames=train_num_frames)


    if test_data == 'OULU_MSU':
        test_data_all = sample_frames(data1 + data4, num_frames=test_num_frames)
    elif test_data == 'CASIA_Idiap':
        test_data_all = sample_frames(data2 + data3, num_frames=test_num_frames)

    data_loader_train = DataLoader(TODDataset(train_data_all, train=True), batch_size=batchsize, shuffle=True,
                                   pin_memory=True,drop_last=True)
    data_loader_test = DataLoader(TODDataset(test_data_all, train=False), batch_size=batchsize, shuffle=True,
                                  pin_memory=True,drop_last=True)
    return data_loader_train, data_loader_test
def set_log(path,train_data,test_data):

    if not os.path.exists(path):
        os.makedirs(path)
    log_path = path + '/Train_' + str(train_data) + '_test_' + str(test_data) + '.txt'
    logger = logging.getLogger("mainModule")
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(filename=log_path, mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger 