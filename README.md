# Airline_sentiment_analysis
This repository performs sentiment analysis on tweets of an airline company by using RNNs. The idea is to know how a user is feeling based 
on his tweet. 

## Requirements 
* tensorfow
* numpy
* pandas
* scikit-learn

*You can install all the requirements with the bellow command*   
pip install -r requirements.txt

## Download pretrained word embedding   
Before doing anything you should first download the pretrained word_embeding from [GloVe](https://nlp.stanford.edu/projects/glove/) by running the following commands
* clone the repository
* cd airline_sentiment_analysis
* run-- > python Download_word_embedding.py

## Run inference
* clone the repository
* cd airline_sentiment_analysis
1- Run Through terminal
* open inference.py edit the text variable
* got to terminal and run --> python inference.py
* This will tell you the sentiment of the sentence
2- Run Through jupyter notebook
* open inference.ipynb 
* edit text variable
* Run the cells to get the sentiment 

## Run training and evaluation
* clone the repository
* cd airline_sentiment_analysis
* open Train_and_evaluate.ipynb
* edit Dataset path and word embeding path if neccessary. (you won't have to edit anything if you kept folders as is)
* Run the cells 

## pipeline
1- explore the dataset:
* Envistage the dataset and knew that the data is skewed( more the 50% of the data is one negative and the rest is either nuetral or 
positive). The steps of the evistagation can be found in this notbook explore_dataset.ipynb   

2- word2vec:
* Used a pretrained word embedding to converts words into vectors that represent each word   

3- Model:
* Built a model consisting of three layers of GRUs followed by on dense layer   

4- Training:
* Because the data skewed I implemented and used the **Stratsified batch**.

#### Accuracy, percision,recall , f1_score and a confusion matrix are all found in this notebook Train_and_evaluate.ipynb after training.
