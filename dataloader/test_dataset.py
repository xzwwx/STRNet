import os

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16, test_clip=10):
        if dataset == 'ucf101':
            self.data_dir = '/data/Disk_B/wucong/UCF_101_output'
        else:
            self.data_dir = '/data/Disk_B/wucong/something-something-v1'

        self.clip_len = clip_len
        self.split = split

        self.resize_height = 128
        self.resize_width = 172
        self.crop_size = 112

        self.test_clip = test_clip

        self.label_array = []

        self.fnames, fnum, labels = [], [], []

        if dataset == 'ucf101':
            if split == 'val':
                split = 'test'
            folder = os.path.join(self.data_dir, split)
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, label)):
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

            self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        elif dataset == 'something_something_v1':
            something_something_v1_list = '/data/Disk_B/wucong/code/IR2+1D/dataloader'
            data_list = os.path.join(something_something_v1_list, '%s_videofolder.txt' % split)
            with open(data_list) as f:
                content = f.readlines()
                for line in content:
                    self.fnames.append(os.path.join(self.data_dir, line.split(' ')[0]))
                    fnum.append(int(line.split(' ')[1]))
                    self.label_array.append(int(line.split(' ')[2].split('\n')[0]))
            f.close()

        print('Number of {} videos of {} dataset: {:d}'.format(split, dataset, len(self.fnames)))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        labels = np.array(self.label_array[index])

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        length = len(frames)
        length_temp = self.test_clip + self.clip_len - 1
        if length < length_temp:
            frame_count = length_temp
        else:
            frame_count = length
        buffer_temp = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        buffer = np.empty((self.test_clip, self.clip_len, self.crop_size, self.crop_size, 3), np.dtype('float32'))

        for i, frame_name in enumerate(frames):
            image = cv2.imread(frame_name)
            if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                image = cv2.resize(image, (self.resize_width, self.resize_height))
            frame = np.array(image).astype(np.float64)
            buffer_temp[i] = frame

        if length < length_temp:
            for i in range(length, length_temp):
                buffer_temp[i] = buffer_temp[length - 1]

        time_temp = (frame_count - self.clip_len + 1) // self.test_clip
        time_index = 0

        height_index = (buffer_temp.shape[1] - self.crop_size) // 2
        width_index = (buffer_temp.shape[2] - self.crop_size) // 2

        for i, clip_data in enumerate(buffer):
            buffer[i] = buffer_temp[time_index:time_index + self.clip_len,
                                    height_index:height_index + self.crop_size,
                                    width_index:width_index + self.crop_size, :]
            time_index += time_temp

        return buffer

    def normalize(self, buffer):
        for i in range(self.test_clip):
            for j, frame in enumerate(buffer[i]):
                frame -= np.array([[[118.84, 118.614, 118.0]]])
                frame /= np.array([[[47.0, 47.1, 47.24]]])
                buffer[i][j] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((0, 4, 1, 2, 3))
