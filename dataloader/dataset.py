import os

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, dataset='ucf101', split='train', clip_len=16):
        if dataset == 'ucf101':
            self.data_dir = ''
        elif dataset == 'something':
            self.data_dir = "/data/Disk_C/something/20bn-something-something-v1/"

        self.clip_len = clip_len #16
        self.split = split

        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = 224

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

        elif dataset == 'something':
            something_something_v1_list = "/data/Disk_A/zhiwei/SomethingCode/"
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

        if self.split == 'train':
            buffer = self.randomcrop(buffer, self.clip_len, self.crop_size)
            buffer = self.randomflip(buffer)
        else:
            buffer = self.centercrop(buffer, self.clip_len, self.crop_size)

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        labels = np.array(self.label_array[index])

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        if len(frames) < 16:
            frame_count = 16
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        for i, frame_name in enumerate(frames):
            image = cv2.imread(frame_name)
            if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                image = cv2.resize(image, (self.resize_width, self.resize_height))
            frame = np.array(image).astype(np.float64)
            buffer[i] = frame

        if len(frames) < 16:
            for i in range(len(frames), 16):
                buffer[i] = buffer[len(frames) - 1]

        return buffer

    def randomcrop(self, buffer, clip_len, crop_size):
        if buffer.shape[0] > clip_len:
            time_index = np.random.randint(buffer.shape[0] - clip_len + 1)
        else:
            time_index = 0

        height_index = np.random.randint(buffer.shape[1] - crop_size + 1)
        width_index = np.random.randint(buffer.shape[2] - crop_size + 1)

        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :]

        return buffer

    def centercrop(self, buffer, clip_len, crop_size):
        if buffer.shape[0] > clip_len:
            time_index = np.random.randint(buffer.shape[0] - clip_len + 1)
        else:
            time_index = 0

        height_index = (buffer.shape[1] - crop_size) // 2
        width_index = (buffer.shape[2] - crop_size) // 2

        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :]

        return buffer

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[118.84, 118.614, 118.0]]])
            frame /= np.array([[[47.0, 47.1, 47.24]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))
