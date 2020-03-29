# Speech-To-Text-STT-
Converting speech or audio files to text in english using Deep Learning
## Description
This project is implemented using Python and Tensorflow(Keras). TensorFlow has released the Speech Commands Datasets. It includes 65,000 one-second long utterances of 30 short words, by thousands of different people. We’ll build a speech recognition system that understands simple spoken commands.
## Usage
### Data-Collection
The original Speech Commands Dataset by Tensorflow can be obtained [here](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)<br>
If you want to visualise an audio file, execute the following commands:<br>
- ```cd Data_Exploration```
- ```python3 Data_Exploration.py```
### Data-Preprocessing
The original dataset has a sampling rate 0f 16kHz and we preprocess it to 8kHz, as the speech frequencies are around 8kHz<br>
![Data-Visualisation of a sample audio file](Images/Visualise.png)<br>
For this purpose, type-in the following commands:<br>
- ```cd Data_PreProcessing```
- ```python3 Process.py```
This process might take from 15-20 minutes depending on the system. For Processed Dataset, click [here](https://drive.google.com/drive/folders/11EePgfin9zqxn8NoY3PQnkiOGzIGOezg?usp=sharing)<br>
### Training
For training, use the following commands:<br>
- ```cd Train_Model```
- ```python3 Train.py```
Before execuing the follwing commands, please make sure to change the corresponding train and validation directories in **Train.py** file.<br>
Training took around 15 minutes on a GTX-1080, i7 processor with 16GB RAM<br>
### Predicting
For predicting or testing out your audio samples, type-in the follwoing command:<br>
- ```cd Prediction```
- ```python3 Predict.py -d path/to/test_audio_file```



