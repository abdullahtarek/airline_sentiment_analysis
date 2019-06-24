# Airline_sentiment_analysis
This repository performs sentiment analysis on tweets of an airline company by using RNNs. The idea is to know how a user is feeling based 
on his tweet. 

## Requirements 
* tensorfow
* numpy
* pandas
* scikit-learn
* urllib
* zipfile


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
1- word2vec:
* Used a pretrained word embedding to converts words into vectors that represent each word   

2- Model:
* Built a model consisting of three layers of GRUs followed by on dense layer   

3- Training:
* Because the data skewed I implemented and used the **Stratsified batch**.

#### Accuracy, percision,recall , f1_score and a confusion matrix are all found in this notebook Train_and_evaluate.ipynb after training.
