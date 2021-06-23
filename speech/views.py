from django.shortcuts import render
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
# Create your views here.
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F950"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def modify_string(msg):
	msg=remove_emoji(msg)
	ps=PorterStemmer()
	i=0
	def remove_punctuation_and_stopwords(s): 
	    l = nltk.word_tokenize(s)
	    l = [x for x in l if not re.fullmatch('[' + string.punctuation + ']+', x)]
	    l=  [word for word in l if word not in stopwords.words('english')]
	    l=  [ps.stem(i) for i in l]
	    return l
	return remove_punctuation_and_stopwords(msg)

def change_to_tfidf(lst):
	l=[' '.join(lst)]
	tfidf=joblib.load('tfidf_vectorizer.sav')
	val = tfidf.transform(l)
	return val

def word_vector(tokens, size):
    model_w2v=joblib.load('model_w2v_model.sav')
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:  # handling the case where the token is not in vocabulary
            continue
    if count != 0:
        vec /= count
    return vec

def handle(msg):
	msg_lst=modify_string(msg)
	wordvec_arrays=np.zeros((1,200))
	wordvec_arrays[0,:]=word_vector(msg_lst,200)
	wordvec_new=pd.DataFrame(wordvec_arrays)
	lst=[change_to_tfidf(msg_lst),wordvec_new]
	return lst	

def home(request):
	if request.method == 'POST':
		s=request.POST.get('message')
		model=load_model('my_model.h5')
		val=handle(s)
		val1=model.predict(val[1])
		color_property='green'
		context={'val':val1,'color_property':color_property,'mess':s,'name':request.POST.get('name'),'script':False}

		return render(request,'speech/home.html',context)
	else:
		color_property='black'
		context={'val':'nothing','color_property':color_property,'mess':None,'name':None,'script':True}
		return render(request,'speech/home.html',context)
