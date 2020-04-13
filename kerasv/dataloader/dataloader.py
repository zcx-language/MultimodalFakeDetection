'''
@Author: your name
@Date: 2020-04-07 12:47:44
@LastEditTime: 2020-04-13 18:27:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \MultimodalFakeDetection\Keras\Dataloader\Dataloader.py
'''

import pathlib
import numpy as np
import librosa
import math
import json
from PIL import Image


class DataLoader:
    ''' load data in training processs
    
    Create train/vail/test data generator
            -- [(batch, time_window, height, width, channel), (batch, time_window, ?, ? ,?)], (batch, label)

    '''
    def __init__(self, batch_size=16, time_window=10, test_rate=0.2, val_rate=0.2, only_video=False):
        super().__init__()
        self.batch_size = batch_size
        self.time_window = time_window
        self.only_video = only_video
        # self.spec_layer = torchaudio.transforms.Spectrogram(n_fft=2048, win_length=2048, hop_length=1470)
        # self.spec_layer = librosa.stft(n_fft=2048, win_length=2048, hop_length=1470)
        
        dataset_parent = pathlib.Path('../data/face/')
        
        dataset_folders = [x for x in dataset_parent.iterdir() if x.is_dir()]
        
        data_folders = []
        for dataset_folder in dataset_folders:
            data_folders.extend([x for x in dataset_folder.iterdir() if x.is_dir()])
        
        data_lens = len(data_folders)
        
        train_size = math.floor(data_lens * (1 - (test_rate + val_rate)))
        test_size = math.floor(data_lens * test_rate)
        
        self.train = data_folders[:train_size]
        self.test = data_folders[train_size:train_size+test_size]
        self.val = data_folders[train_size+test_size:]
        
        print("Data split into train:{} | test:{} | val:{}".format(len(self.train), len(self.test), len(self.val)))
    
    def get_data_len(self):
        return (len(self.train), len(self.val), len(self.test))

    def generator(self, train=True, val=False, no_timewindow=False):
        t_gen = self.get_timewindow(train, val, no_timewindow)
        
        while True:
            batch_Xv, batch_Xa, batch_Y = [], [], []

            while len(batch_Xv) < self.batch_size:
                Xv, Xa, Y = next(t_gen)
                batch_Xv.append(Xv)
                batch_Xa.append(Xa)
                batch_Y.append(Y)

#             print('spitting out: {} | {} | {}'.format(np.shape(batch_Xv), np.shape(batch_Xa), np.shape(batch_Y)))

            if self.only_video:
                yield np.array(batch_Xv), np.array(batch_Y)
            else:
                yield [np.array(batch_Xv), np.array(batch_Xa)], np.array(batch_Y)


    def get_timewindow(self, train=True, val=False, no_timewindow=False):
        Xv, Xa, Y = [],[],[]

        sub_data_folders = self.train
        if not train:
            if not val:
                sub_data_folders = self.test
            else:
                sub_data_folders = self.val
        
        while True:
            for sub_data_folder in sub_data_folders:
    
                json_path = list(sub_data_folder.parent.glob('*.json'))[0]
                with open(json_path) as json_file:
                    json_data = json.load(json_file)
                label = json_data[sub_data_folder.name+'.mp4']['label']
                
                if label == 'FAKE':
                    label = 1
                else:
                    label = 0
                
                wav_path = list(sub_data_folder.glob('audio.wav'))[0]
                # wav, sr = torchaudio.load(wav_path)
                wav, sr = librosa.load(wav_path)
                specgram_frames = librosa.stft(y=wav, n_fft=2048, win_length=2048, hop_length=1470).permute(2,1,0).reshape(-1,25,41,2)

                face_imgs = [x for x in sub_data_folder.glob('*.jpg') if x.stem.find('_') == -1] 
                face_img_nums = len(face_imgs)

                get_nums, i = 0, 0
                while(get_nums < self.time_window and i < face_img_nums):
                    try:
                        Xv.append(np.array(Image.open(face_imgs[i])))
                        Xa.append(specgram_frames[i].numpy())
                        get_nums  = get_nums + 1
                    except Exception as e:
                        pass
#                         print(e, 'in frame{}-{}'.format(i, str(f_name)))
                    finally:
                        i = i + 1
                Y.append(label)

                if(len(Xv) == self.time_window):
                    yield Xv, Xa, Y
                    
                Xv, Xa, Y = [],[],[]
    
    