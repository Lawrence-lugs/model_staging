#%%

import torchaudio 
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
import torch
import sys
from cachetools.func import lru_cache

# ToyCar sounds are at 16kHz sample rate
root = '/home/raimarc/lawrence-workspace/data/ToyCar/train/'

def get_audio_features(filepath):
    waveform, sample_rate = torchaudio.load(filepath)

    #print(filepath)
    #print(waveform.shape)

    n_fft = 1024
    win_length = n_fft 
    hop_length = win_length // 2

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate,n_fft=n_fft,win_length=1024,hop_length=hop_length)
    mel_spectrogram = mel_transform(waveform)

    log_mel_energies = 10 * torch.log10(mel_spectrogram + sys.float_info.epsilon)
    nearest_5_multiple = log_mel_energies.shape[-1] - log_mel_energies.shape[-1]%5 
    log_mel_energies = log_mel_energies[:,:,:nearest_5_multiple]

    # black magic reshaping workaround
    f1 = log_mel_energies.squeeze().permute(1,0)
    f2 = f1.reshape(5,-1,128).permute(1,0,2)
    f3 = f2.reshape(-1,640)

    std=10.885
    mean=-10.6091
    f4 = (f3 - mean)/std

    return f4

@lru_cache(1)
def get_audio_features_from_wav(waveform, sample_rate = 16000,crop=True,normalize=True):
    if crop:
        waveform = waveform[:,31680:144320]

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate,
                                            n_mels=128,
                                            n_fft=1024,
                                            win_length=1024,
                                            hop_length=512,
                                            power=2,
                                            mel_scale='slaney',
                                            norm = 'slaney',
                                            pad_mode = 'constant',
                                            center = True
                                            )
    mel_spectrogram = mel_transform(waveform)

    log_mel_energies = 20.0 / 2.0 * torch.log10(mel_spectrogram + sys.float_info.epsilon)
    f1 = log_mel_energies.squeeze().permute(1,0) # (t,128)

    f2 = []
    for t,feat in enumerate(f1):
        f2.append(f1[t:t+5].flatten())
    f3 = torch.stack(f2[:-4])

    # Measured over the training set 
    std = 12.0590
    mean = -28.0861
    if normalize:
        f3 = (f3 - mean)/std

    return f3

@lru_cache(1)
def get_audio_features_librosa_based(filepath, crop=True, normalize=True):
    waveform, sample_rate = torchaudio.load(filepath)

    if crop:
        waveform = waveform[:,31680:144320]

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate,
                                            n_mels=128,
                                            n_fft=1024,
                                            win_length=1024,
                                            hop_length=512,
                                            power=2,
                                            mel_scale='slaney',
                                            norm = 'slaney',
                                            pad_mode = 'constant',
                                            center = True
                                            )
    mel_spectrogram = mel_transform(waveform)

    log_mel_energies = 20.0 / 2.0 * torch.log10(mel_spectrogram + sys.float_info.epsilon)
    f1 = log_mel_energies.squeeze().permute(1,0) # (t,128)

    f2 = []
    for t,feat in enumerate(f1):
        f2.append(f1[t:t+5].flatten())
    f3 = torch.stack(f2[:-4])

    # Measured over the training set 
    std = 12.0590
    mean = -28.0861
    if normalize:
        f3 = (f3 - mean)/std

    return f3

def librosa_get_audio_features(filepath,n_mels=128,frames=5,n_fft=1024,hop_length=512,power=2.0):
    
    import librosa
    import numpy as np

    dims = n_mels * frames

    y, sr = librosa.load(filepath,sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    if vector_array_size < 1:
        return np.empty((0, dims))

    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return torch.Tensor(vector_array)

def get_mel_dataset(root, set = 'train',loader = get_audio_features_librosa_based,part=None,use_tqdm=False):
    import os
    from tqdm import tqdm

    root = root + f'{set}/'

    list = []

    if part is None:
        files = os.listdir(root)
    else:
        files = os.listdir(root)[part[0]:part[1]]

    iterable = files
    if use_tqdm:
        iterable = tqdm(files,desc=f'Loading ToyCar {set}set')

    for filename in iterable:
        mel_bins = loader(root+filename)
        if set == 'test':
            if filename[0] == 'a':
                lbl = 0
            else:
                lbl = 1
            list.append((mel_bins,lbl))
        if set == 'train':
            for bin in mel_bins:
                list.append(bin)
    return list

def get_wav_dataset(root, set = 'train', start = None, end = None, use_tqdm=False, files = None):
    import os
    from tqdm import tqdm

    root = root + f'{set}/'

    if files is None:
        if start is None:
            files = os.listdir(root)
        else:
            files = os.listdir(root)[start:end]

    list = []
    iterable = files
    if use_tqdm:
        iterable = tqdm(files,desc=f'Loading ToyCar {set}set')

    for filename in iterable:
        wav, sr = torchaudio.load(root+filename)
        if set == 'test':
            if filename[0] == 'a':
                lbl = 0
            else:
                lbl = 1
            list.append((wav,lbl))
        if set == 'train':
            list.append(wav)
    return list

class toycar_dataset(Dataset):
    '''
    ToyCar machine subset of DCASE2020 Task 2 Dataset

    Train set is a dataset of 1D vectors of size 640 for an autoencoder

    Test set is a dataset of 2D vectors of size t,640 depending on the length of input wav
    '''
    def __init__(self,transform=None, set='train', type='wav', root = '/home/raimarc/lawrence-workspace/data/ToyCar/', blocks = True):
        self.transform = transform
        self.type = type
        self.set = set
        
        testfile = root + 'train/normal_id_01_00000000.wav'
        self.n_data_in_wav = get_audio_features_librosa_based(testfile).shape[0]
        self.block = None
        self.root = root + f'{set}/'
        self._root = root

        from os import listdir
        self.files = listdir(self.root)

        # if type == 'mel':
        #     self.protoset = get_mel_dataset(root=root, set=set)
        # if type == 'wav':
        #     self.protoset = get_wav_dataset(root=root, set=set)

    def __getitem__(self,idx):

        if self.type == 'mel':
            data = self.protoset[idx]
            return data
        
        if self.type == 'wav' and self.set == 'train':

            wav_idx = idx // self.n_data_in_wav
            data_idx = idx % self.n_data_in_wav

            # return get_audio_features_librosa_based(root + self.files[wav_idx])[data_idx]

            BLOCKSIZE = 2000

            block_idx = wav_idx // BLOCKSIZE
            if self.block != block_idx:
                # print(f'Getting Trainset Block {block_idx}...')
                self.protoset = get_wav_dataset(root=self._root,set=self.set,start = block_idx*BLOCKSIZE, end = (block_idx+1)*BLOCKSIZE, use_tqdm=False, files = self.files)
                self.block = block_idx
            data = self.protoset[wav_idx % BLOCKSIZE]
            return get_audio_features_from_wav(data)[data_idx]

        if self.type == 'wav' and self.set == 'test':

            # return get_audio_features_librosa_based(root + self.files[idx])

            if idx >= len(self):
                raise StopIteration

            BLOCKSIZE = 5000

            block_idx = idx // BLOCKSIZE
            if self.block != block_idx:
                # print(f'Getting Testset Block {block_idx}...')
                self.protoset = get_wav_dataset(root=self._root,set=self.set,start = block_idx*BLOCKSIZE, end = (block_idx+1)*BLOCKSIZE, use_tqdm=False, files = self.files)
                self.block = block_idx
            wav, lbl = self.protoset[idx % BLOCKSIZE]
            return get_audio_features_from_wav(wav),lbl

    def __len__(self):
        length = len(self.files)
        if self.set == 'train' and self.type!='mel':
            length *= self.n_data_in_wav
        return length

# %%

if __name__ == '__main__':
    my_set = toycar_dataset(set='test',type='wav')
    print(len(my_set))

# %%
