import json
import pathlib
import pickle
import shutil

# import torchaudio
import librosa
import numpy as np
import torch
from facenet_pytorch import MTCNN
from moviepy.editor import VideoFileClip
from PIL import Image

import cv2

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')



class ExtractFaceAudio:
    '''
    class for detecting faces in the frames and extracting audio  of a video file.
    '''
    def __init__(self, detector, n_frames=None, batch_size=60, resize=None):
        '''Constructor for ExtractFaceAudio class.
        
        Keyword Arguments:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            batch_size {int} -- Batch size to use with MTCNN face detector. (default: {32})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        '''
        self.detector = detector
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.resize = resize
    
    def __call__(self, video_path, face_folder):
        """Load frames from an MP4 video and detect faces.

        Arguments:
            video_path {pathlib.Path} -- Path to video.
            face_folder {pathlib.Path} -- Path to save face image and video audio
        """

        # Extract audio from video
        v_clip = VideoFileClip(video_path)
        audio = v_clip.audio
        wav_path = video_path.replace('.mp4', '.wav')
        audio.write_audiofile(wav_path)
        
        if face_folder:
            audio.write_audiofile(face_folder+'/audio.wav')
        
        # wav, sr = torchaudio.load(wav_path)
        wav, sr = librosa.load(wav_path, mono=True)

        # Create video reader and find length
        v_cap = cv2.VideoCapture(video_path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        faces = []
        frames = []
        face_paths = []     # face image file name list - corresponding to the frame number
        
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])
                
                frames.append(np.asarray(frame))
                face_paths.append(face_folder + '/' + str(j) + '.jpg')
                
                # When batch is full, detect faces and reset frame list
                if len(frames) % self.batch_size == 0 or j == sample[-1]:
                    if face_folder:
                        faces.extend(self.detector(frames, save_path=face_paths))
                    else:
                        faces.extend(self.detector(frames))
                        
                    frames = []
                    face_paths = []

        v_cap.release()

        return faces, wav


def Preprocessing(extractor=None, dfdc_parent=None, pickle_parent=None, face_parent=None):
    '''get data from video files
    
    Arguments:
        dfdc_parent {pathlib.Path} -- data file parent
        pickle-parent {pathlib.Path} -- data file parent
        face_parent {pathlib.Path} -- face file parent
    '''
    
    orig_video_num = 0
    fake_video_num = 0
    
    dfdc_folders = [x for x in dfdc_parent.iterdir() if x.is_dir()]

    for dfdc_folder in dfdc_folders:
        
        video_paths = dfdc_folder.glob('*.mp4')
        json_file = list(dfdc_folder.glob('*.json'))[0]

        if pickle_parent:
            pickle_folder = pickle_parent / dfdc_folder.name
            pickle_folder.mkdir(parents=True, exist_ok=True)
        
        if face_parent:
            face_folder = face_parent / dfdc_folder.name
            face_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy(str(json_file), str(face_folder.absolute()))
        
        with open(json_file) as f:
            json_data = json.load(f)
                
        for video_path in video_paths:
            
            video_face_folder = None
            if face_parent:
                video_face_folder = face_folder / video_path.stem
                video_face_folder.mkdir(exist_ok=True)
                        
            # Get videos's label from the json file
            video_label = 0
            label = json_data[str(video_path.name)]['label']
            if label == 'FAKE':
#                 continue    # get more original video to fix the orig-fake rate of the train set
                video_label = 1
                fake_video_num = fake_video_num + 1
            else:
                orig_video_num = orig_video_num + 1
            
            
            # Judge this video whether processed or not
            if pickle_parent:
                pickle_file_path = pickle_folder / (video_path.stem + '.pickle')
                if pickle_file_path.exists():
                    print(str(pickle_file_path), 'is existed, so skipped..')
                    continue
            
                    
            if pickle_parent or face_parent:
                try:
                    faces, wav = extractor(str(video_path), str(video_face_folder))
                except Exception as e:
                    print(e)
                    continue

            # show detected faces
    #         for face in faces:
    #             if face is not None:
    #                 img = face.byte()
    #                 img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    #                 plt.imshow(Image.fromarray(img))
    #                 plt.pause(0.01)
    #             break

            # save to pickle file
            if pickle_parent:
                pickle_file = open(str(pickle_file_path), 'wb')
                pickle.dump((faces.numpy(), wav.numpy(), video_label), pickle_file)
                pickle_file.close()
            print('\u255f {} was done.'.format(str(video_path)))
#             break # remove this to do all files
        break # remove this to do all folders
        
    print('original video number:{}, fake video number:{}'.format(orig_video_num, fake_video_num))
    return


def main():
    mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()
    extractor = ExtractFaceAudio(detector=mtcnn, batch_size=60, resize=0.25)
    
    dfdc_parent = pathlib.Path('E:/Database/DFDC/')
    pickle_parent = None    #pathlib.Path('/nas/data/zcx/pickle/')
    face_parent = pathlib.Path('../Data/face/')
    
    Preprocessing(extractor, dfdc_parent, pickle_parent, face_parent)


if __name__ == '__main__':
    main()