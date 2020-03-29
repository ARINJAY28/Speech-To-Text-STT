# Speech-To-Text-STT-
Converting speech or audio files to text in english using Deep Learning
## Description
This project is implemented using Python and Tensorflow(Keras). TensorFlow has released the Speech Commands Datasets. It includes 65,000 one-second long utterances of 30 short words, by thousands of different people. We’ll build a speech recognition system that understands simple spoken commands.
## Usage
### Data-Collection
The original Speech Commands Dataset by Tensorflow can be obtained [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)<br>
### Data-Preprocessing
The original dataset has a sampling rate 0f 16kHz and we preprocess it to 8kHz, as the speech frequencies are around 8kHz<br>
![Data-Visualisation of a sample audio file](Images/Visualise.png)<br>
For this purpose, type-in the following commands:<br>
'''cd Data_PreProcessing'''
'''python3 Process.py'''
