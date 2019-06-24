import numpy as np

def read_word2vec(file_path):
    word_embedding={}
    with open(file_path,'r') as word_embedding_file:
        linenum=0
        for line in word_embedding_file:
            
            words = line.split(" ")
            words[-1]=words[-1].replace('\n',"")
            word_embedding[words[0]] =np.array(words[1:], dtype=np.float16)
    return word_embedding

def convert_to_vector(text ,word_embedding, maxSequence):
    wordembedding_len= list(word_embedding.values())[0].shape[0]
    wordVec=np.zeros((maxSequence, wordembedding_len),dtype=np.float16)
    words = text.split(' ')
    wordVec_index = maxSequence-1
    for index in range(len(words)-1,-1,-1):
        if words[index] in word_embedding:
            wordVec[wordVec_index]= word_embedding[words[index]]
        else:
            wordVec[wordVec_index]= list(word_embedding.values())[len(word_embedding)-1]
        wordVec_index-=1
    return wordVec