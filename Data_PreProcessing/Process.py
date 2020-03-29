#importing libraries
import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from progressbar import ProgressBar
from tqdm import tqdm


from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
K.clear_session()


warnings.filterwarnings("ignore")

'''In the dataset, most of the speech related frequencies are around 8000Hz. Hence we resample it to 8000Hz'''
#train audio path
train_audio_path = '../Data/train/audio/'

labels=os.listdir(train_audio_path)
pbar = ProgressBar()

#processing the audio files
print("=========================================================================================================================")
print("Processing Audio Files")
print("=========================================================================================================================")
all_wave = []
all_label = []
for label in tqdm(labels):
	print("Label : " , label)
	waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
	for wav in waves:
		samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
		samples = librosa.resample(samples, sample_rate, 8000)
		File_Name = "../Data/Resampled/train/" + label + '/' + wav
		if(len(samples)== 8000) :
			librosa.output.write_wav(File_Name,samples,8000)
			all_wave.append(samples)
			all_label.append(label)

print("Labels encountered in training directory")
print(all_label)
print("=========================================================================================================================")

