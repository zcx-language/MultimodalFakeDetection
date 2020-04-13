'''
@Author: your name
@Date: 2020-04-12 11:43:17
@LastEditTime: 2020-04-13 18:27:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \MultimodalFakeDetection\Keras\main.py
'''

import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard

from dataloader.dataloader import DataLoader
from networks.video_audio_model import create_model


def configDevice(devices='0'):
    config = tf.ConfigProto()
    # 指定可见显卡
    config.gpu_options.visible_device_list = devices
    #不满显存, 自适应分配
    config.gpu_options.allow_growth = True   
    sess = tf.Session(config=config)
    KTF.set_session(sess)
    return


def main():
   
    configDevice()
   
    time_window = 16
    video_w = 160
    video_h = 160
    video_c = 3

    audio_l = 1024
    audio_c = 2

    n_epochs = 10

    G = DataLoader(batch_size=16, time_window=time_window, only_video=True)
    train_data_len, vali_data_len, test_data_len = G.get_data_len()

    train_generator = G.generator()
    test_generator = G.generator(False)
    validation_generator = G.generator(False, True)

    model = create_model((time_window, video_h, video_w, video_c), (time_window, 25, 41, 2))

    class_weight = {0:14.6, 1:1.}
    model.fit_generator(train_generator, steps_per_epoch=train_data_len//n_epochs, verbose=1, epochs=n_epochs, validation_data=validation_generator, validation_steps=vali_data_len//n_epochs, 
                            class_weight=class_weight, callbacks=[TensorBoard(log_dir='./logs/multimodal/')])
                            
    test_result = model.evaluate_generator(test_generator, steps=test_data_len//n_epochs)

    print(test_result)
    model.save('./models/multimodel.h5')
    return


if __name__ == '__main__':
    main()
