import os
import speech_recognition as sr
from tqdm import tqdm
import moviepy.editor as mp
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
from textblob import TextBlob
import subprocess

#nltk.download('corpus')

r = sr.Recognizer()
# Step 1 : Convert Video to Text
#clip = mp.VideoFileClip("lion-sample.wmv").subclip(100,116)
clip = mp.VideoFileClip("lion-sample.wmv")
clip.audio.write_audiofile("theaudio.wav")

Audio = sr.AudioFile('theaudio.wav')
with Audio as source:
    Audio = r.record(source)
    try:
        Text = r.recognize_google(Audio)
        #Text = r.recognize_sphinx(Audio, language="en-US")
        #print(Text)
        file = open('testfile.txt',"w")  
        file.write(Text)
        file.close() 
    except Exception as e:
        print(e)

# Step 2 : Train Text
class Articleprocessor:
    
#    def __init__(self):
#        pass
##        self.page = title
##        self.summary = TextBlob(self.page.summary)

    def tokenize(self,sent):
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def parse_stories(self, lines):
        corpus = []
        ReadFile = re.sub('[^a-zA-Z]', ' ', lines)
        ReadFile = ReadFile.lower()
        ReadFile = ReadFile.split()
        #    ReadFile  = [word for word in ReadFile if word not in set(stopwords.words('english'))]
        ps = PorterStemmer()
        ReadFile = [ps.stem(word) for word in ReadFile if not word in set(stopwords.words('english'))]
        ReadFile = ' '.join(ReadFile)
        corpus.append(ReadFile)
        return corpus

    def get_stories(self,f):
        try:
            f = open(f,'r')
            Text_Message = f.read()
            f.close()
#           from nltk.tokenize import sent_tokenize
#           sent_tokenize_list = sent_tokenize(Text_Message)
            #print(sent_tokenize_list)
            stories = ArticleClass.parse_stories(Text_Message)
            return stories
        except Exception as e:
            print(e)

# =============================================================================
ArticleClass = Articleprocessor()
X_Train = ArticleClass.get_stories('testfile.txt')

X_test = input('Please enter your Query:')
X_test_After_parsing = ArticleClass.parse_stories(X_test)
 X_test = tokenize(X_test)
# =============================================================================



# Creating the Bag of Words model
# =============================================================================
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = 1500)
# X = cv.fit_transform(corpus).toarray()
# y = dataset.iloc[:, 1].values
# 
# 
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# 
# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
# cm = confusion_matrix(y_test, y_pred)
# ac = accuracy_score(y_test, y_pred)
# print(classification_report(y_test, y_pred))
# =============================================================================




    


        

    
    
    
        
        
        
        
        
        
        
    
    
    