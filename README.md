# Deep learning in the classification of musical songs - data preprocessing.

### Scope of work

The work includes an analysis of preprocessing methods possible for use in the classification of musical songs. Classification is limited here to methods based on deep learning. Tests to determine the quality of classification, for different deep learning methods and different data preprocessing methods, are to be carried out using the Python language and databases GTZAN and FMA dataset.

## Setup Virtualenv on Windows

```console
deactivate
rmdir venv
py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install virtualenv
py -3.10 -m virtualenv venv
.\venv\Scripts\activate
.\venv\Scripts\python.exe -m pip install -r .\requirements\requirements.txt
```

If any problem with activating venv try: 

```
For Windows 11, Windows 10, Windows 7, Windows 8, Windows Server 2008 R2 or Windows Server 2012, run the following commands as Administrator:

x86 (32 bit)
Open C:\Windows\SysWOW64\cmd.exe
Run the command: powershell Set-ExecutionPolicy RemoteSigned

x64 (64 bit)
Open C:\Windows\system32\cmd.exe
Run the command: powershell Set-ExecutionPolicy RemoteSigned
```

## Datasets

### GTZAN Dataset - Music Genre Classification

Source: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

#### Context

Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.
This data hopefully can give the opportunity to do just that.

#### Content
- **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
- **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
- **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.

#### Acknowledgements

- The GTZAN dataset is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions (http://marsyas.info/downloads/datasets.html).
- This was a team project for uni, so the effort in creating the images and features wasn't only my own. So, I want to thank James Wiltshire, Lauren O'Hare and Minyu Lei for being the best teammates ever and for having so much fun and learning so much during the 3 days we worked on this.