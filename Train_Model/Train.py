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
train_audio_path = '../Data/Resampled/train/'

labels=os.listdir(train_audio_path)
pbar = ProgressBar()

#Loading the audio files
print("=========================================================================================================================")
print("Loading Audio Files")
print("=========================================================================================================================")
all_wave = []
all_label = []
for label in tqdm(labels):
	waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
	for wav in waves:
		samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
		samples = librosa.resample(samples, sample_rate, 8000)
		all_wave.append(samples)
		all_label.append(label)


print("=========================================================================================================================")


#Integer encoding the labels
le = LabelEncoder()
y=le.fit_transform(all_label)
classes= list(le.classes_)

#one-hot encoiding the labels since we have a multi-class classification problem
y=np_utils.to_categorical(y, num_classes=len(labels))

#Reshaping input from 2d to 3d
all_wave = np.array(all_wave).reshape(-1,8000,1)

#train-test-split the dataset
x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)

#model architecture
inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
print("Model Summary")
print(model.summary())
print("=========================================================================================================================")

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#fitting the model
history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

#Diagnostic-Plot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()