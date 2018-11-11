# Real Time Face Recognition using Keras (Tensorflow Backend)
This is a small project on Real time face recognition using `Python 3.6.3` `Keras` and `Open-CV`. Siamese Network is used as the basis for the recognition. 

## Inspiration
This implementation is heavily inspired by VGG-Face-Net.

## Pre-processing
Open-CV is used for the preprocessing of the captured face. More specifically, the Haar-Cascade classifier is used for detection and alignment.

## Running
To add a new face to the database, run: `AddToDatabase.py`

To remove a face from the database, run: `RemoveFromDatabase.py`

To execute the project, run: `FR_driver.py`

## Results
![Result1](/results/Capture.PNG)
![Result2](/results/Capture2.PNG)
