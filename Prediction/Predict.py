import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from pydub import AudioSegment
from pydub.playback import play
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the directory of indexed images")
args = vars(ap.parse_args())

print("=====================================================================================")
Audio_File = args["dataset"]

from keras.models import load_model
model=load_model('../Train_Model/best_model.hdf5')

train_audio_path = '../Data/train/audio'
labels=os.listdir(train_audio_path)

le = LabelEncoder()
y=le.fit_transform(labels)
classes= list(le.classes_)


y=np_utils.to_categorical(y, num_classes=len(labels))

def predict(audio,classes):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]



#reading the voice commands
samples, sample_rate = librosa.load(Audio_File, sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)

#block to play the test audio file
print("=====================================================================================")
print("Playing Test Audio File")
song = AudioSegment.from_wav(Audio_File)
play(song)   
print("=====================================================================================")         

#converting voice commands to text
print("=====================================================================================")
print("Prediction:")
print(predict(samples,classes))
print("=====================================================================================")