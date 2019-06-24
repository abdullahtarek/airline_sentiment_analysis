from word2vec import convert_to_vector
import numpy as np

def one_hot(sentiment):
    #0 neutral
    #1 positive
    #2 negative
    if sentiment == "neutral":
        onehot_arr=[1,0,0]
    elif sentiment == "positive":
        onehot_arr=[0,1,0]
    else:
        onehot_arr=[0,0,1]
    return onehot_arr

def Make_Dataset(csv_file,word_embedding,maxSequence):
    #shuffle Dataset
    csv_file = csv_file.iloc[np.random.permutation(len(csv_file))]
    input_x=[]
    GT_y=[]
    for line in csv_file.iterrows():
        text = convert_to_vector(line[1]['text'], word_embedding,maxSequence )
        airline_sentiment = one_hot(line[1]['airline_sentiment'])
        
        input_x.append(text)
        GT_y.append(airline_sentiment)
        
    input_x=np.array(input_x)
    GT_y=np.array(GT_y)
    return input_x , GT_y

def train_test_split(data , labels , train_percentage):
    train_data   =  data[0:int(len(data)*0.8)]
    train_labels =  labels[0:int(len(labels)*0.8)]

    val_data   =  data[int(len(data)*0.8):]
    val_labels =  labels[int(len(labels)*0.8):]
    
    return train_data,train_labels,val_data,val_labels