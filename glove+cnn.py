# -*- coding: utf-8 -*-

import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump,load
import numpy as np
import torch.nn as nn
import torch
import sys
import math
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

def load_doc(filename):
    file=open(filename,'r')
    text=file.read()
    file.close()
    return text

def clean_doc(doc):
    tokens=doc.split()
    re_punc=re.compile('[%s]'%re.escape(string.punctuation))
    tokens=[re_punc.sub('',w) for w in tokens]
    tokens=[word for word in tokens if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    tokens=[w for w in tokens if not w in stop_words]
    tokens=[word for word in tokens if len(word) > 1]
    #tokens=' '.join(tokens)
    return tokens

def process_docs(directory,is_train):
    documents=list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path=directory+'/'+filename
        doc=load_doc(path)
        tokens=clean_doc(doc)
        documents.append(tokens)
    return documents

def load_clean_dataset(is_train):
    neg=process_docs('txt_sentoken/neg',is_train)
    pos=process_docs('txt_sentoken/pos',is_train)
    docs=neg+pos
    labels=[0 for _ in range(len(neg))]+[1 for _ in range(len(pos))]
    return docs,labels

def save_dataset(dataset,filename):
    dump(dataset,open(filename,'wb'))
    print('Saved: %s'%filename)

train_docs,ytrain=load_clean_dataset(True)
test_docs,ytest=load_clean_dataset(False)
save_dataset([train_docs,ytrain],'train.pkl')
save_dataset([test_docs,ytest],'test.pkl')

class Vocab:
    def __init__(self,train_docs,test_docs):
        self.words=['pad']
        self.word_vecs=None
        self.pad=0
        self.length=0
        for i in train_docs:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        for i in test_docs:
            for j in i:
                if j not in self.words:
                    self.words.append(j)
        reverse=lambda x: dict(zip(x,range(len(x))))
        self._word2id=reverse(self.words)#字典word:id
        self.length=len(self.words)
        self.inGoogle=[]
        self.word_vecs=np.zeros((self.length,100))
    
    def words2indices(self,sents):
        return [[self._word2id[w] for w in s] for s in sents]
        
    def load_bin_vec(self,fname):
        with open(fname) as fp:
            for line in fp.readlines():
                line=line.split(" ")
                word=line[0]
                if word in self.words:
                    self.word_vecs[self._word2id[word]]=np.array([float(x) for x in line[1:]])
                    self.inGoogle.append(word)

    def add_unknown_words(self,word_vecs,k=100):
        for word in self.words:
            if word not in self.inGoogle:
                self.word_vecs[self._word2id[word]]=np.random.uniform(-0.25,0.25,k)
    
    def pad_sents(self,sents,pad_token):#(batch_size,max_length)
        sents_padded=[]
        max_length=max([len(s) for s in sents])
        for i in sents:
            data=i
            data.extend([pad_token for _ in range(max_length-len(i))])
            sents_padded.append(data)
        return sents_padded
    
    def to_input_tensor(self,sents,device):
        wordIndices=self.words2indices(sents)
        wordpad=self.pad_sents(wordIndices,self._word2id['pad'])
        #(batch_size,max_length)
        wordten=torch.tensor(wordpad,dtype=torch.long,device=device)
        return wordten

vectors_file='../../data/glove.6B.100d.txt'
vocab=Vocab(train_docs,test_docs)
vocab.load_bin_vec(vectors_file)

class Model(nn.Module):
    def __init__(self,vocab,device):
        super(Model,self).__init__()
        self.device=device
        weight=torch.FloatTensor(vocab.word_vecs).to(device)
        self.embedding_static=nn.Embedding.from_pretrained(weight)
        self.embedding_static.requires_grad=False
        self.embedding_change=nn.Embedding.from_pretrained(weight)
        self.embedding_change.requires_grad=True
        
        self.convd3=nn.Conv1d(in_channels=100,out_channels=1,kernel_size=3,padding=1)
        self.convd4=nn.Conv1d(in_channels=100,out_channels=1,kernel_size=4,padding=1)
        self.convd5=nn.Conv1d(in_channels=100,out_channels=1,kernel_size=5,padding=1)
        self.linear=nn.Linear(3,2)
        self.dropout=nn.Dropout(0.5)
        self.vocab=vocab
        
    
    def forward(self,source):
        source_lengths=[len(s) for s in source]
        source_padded=self.vocab.to_input_tensor(source,self.device)
        #print('source_padded:',source_padded)
        #print('source_padded.shape:',source_padded.shape)
        source_embedding_static=self.embedding_static(source_padded).permute(0,2,1)
        source_embedding_change=self.embedding_change(source_padded).permute(0,2,1)
        #print('source_embedding_change.shape:',source_embedding_change.shape)#(batch_size,max_length,embedding_dim)
        conv_out3_1=self.convd3(source_embedding_static)
        conv_out4_1=self.convd4(source_embedding_static)
        conv_out5_1=self.convd5(source_embedding_static)
        #print('conv_out1.shape:',conv_out3_1.shape)
        conv_out3_2=self.convd3(source_embedding_change)
        conv_out4_2=self.convd4(source_embedding_change)
        conv_out5_2=self.convd5(source_embedding_change)
        #print('conv_out3_2.shape:',conv_out3_2.shape)
        conv_out3=torch.cat((conv_out3_1,conv_out3_2),2)
        conv_out4=torch.cat((conv_out4_1,conv_out4_2),2)
        conv_out5=torch.cat((conv_out5_1,conv_out5_2),2)
        #print('conv_out3.shape:',conv_out3.shape)
        conv_out3_max=torch.max(conv_out3,dim=2).values
        conv_out4_max=torch.max(conv_out4,dim=2).values
        conv_out5_max=torch.max(conv_out5,dim=2).values
        conv_out_max=torch.cat((conv_out3_max,conv_out4_max,conv_out5_max),1)
        #print('conv_out_max.shape:',conv_out_max.shape)
        #print('conv_out_max:',conv_out_max)
        #drop_out=self.dropout(conv_out_max)
        result=self.linear(conv_out_max)
        result=F.softmax(result)
        #print('result:',result)
        #print('result.shape:',result.shape)
        return result

def batch_iter(data,batch_size,shuffle=True):
    #data中包括样本和标签
    batch_num=math.ceil(len(data)/batch_size)
    index_array=list(range(len(data)))
    if shuffle:
        np.random.shuffle(index_array)
    for i in range(batch_num):
        indices=index_array[i*batch_size:(i+1)*batch_size]
        example=[data[idx] for idx in indices]
        example=sorted(example,key=lambda e:len(e[0]),reverse=True)
        sents=[e[0] for e in example]
        labels=[e[1] for e in example]
        yield sents,labels

from torch.nn import init
#define the initial function to init the layer's parameters for the network
def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

epochs=100
batch_size=10
def train():
    #device=torch.device("cuda:0" if args['--cuda'] else "cpu")
    device=torch.device("cuda:0")
    print('use device: %s' % device,file=sys.stderr)
    model=Model(vocab,device)
    model.apply(weigth_init)
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    train_data=list(zip(train_docs,ytrain))
    test_data=list(zip(test_docs,ytest))
    model=model.to(device)
    for i in range(epochs):
        allLoss=0
        for sents,labels in batch_iter(train_data,batch_size):
            labels=torch.LongTensor(labels).to(device)
            optimizer.zero_grad()
            new_batch_size=len(sents)
            result=model(sents)
            entrycross=nn.CrossEntropyLoss()
            #print('result:',result)
            #print('labels:',labels)
            loss=entrycross(result,labels)
            #print('loss:',loss)
            allLoss+=loss
            loss.backward()
            optimizer.step()
        print('[INFO] epoch:{} loss: {}'.format(i,allLoss))
    torch.save(model,'model.pkl')

def test():
    model=torch.load('model.pkl')
    device=torch.device("cuda:0")
    print('use device: %s' % device,file=sys.stderr)
    model=model.to(device)
    with torch.no_grad():
        prob=model(test_docs)
        result=[]
        for i in range(prob.shape[0]):
            if prob[i,1]>=0.5:
                result.append(1)
            else:
                result.append(0)
        print('accuracy:{}'.format(accuracy_score(ytest,result)))

train()

test()

