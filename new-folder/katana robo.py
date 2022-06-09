#!/usr/bin/env python
# coding: utf-8

# In[1]:



# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import random


# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# In[4]:


dise=pd.read_csv('D:/data/Training.csv')
dis = pd.DataFrame(dise)


# In[5]:


def find_sym():
    k=dis['prognosis'].unique()
    while True:
        print('Which Disease Symtom you want to know? Enter its Index Number.')
        for i in range(6):
            print(i+1,'-',k[i])
        print('7 - Other')
        s=int(input())
        if(s!=7):
            return k[s-1]
        else:
            k=k[7:]


# In[6]:


def sym(s):
    p=dis[dis['prognosis']==s]
    col=dis.columns
    col=col[:-1]
    dise=[]
    for i in col:
        if p[i].max()==1:
            dise.append(i)    
    print('Symptoms of',s,'are :-')
    for i in range(len(dise)):
        print('>>',dise[i])
        if(i>6):
            break


# In[7]:


emotion=pd.read_json('D:/data/quotes.json')
emotion=emotion.drop(['Author','Tags'],axis=1)
categories = emotion['Category']
cat = categories.unique()


# In[8]:


def emot(s):
    fo=pd.DataFrame(emotion.loc[emotion['Category']==s])
    fo=fo.sort_values('Popularity',ascending=False)
    i=random.randrange(0,50)
    return fo.iloc[i].Quote


# In[9]:


data=pd.read_csv('D:/data/bmi.csv')

gen=pd.get_dummies(data['Gender'])

data=pd.concat([data,gen],axis=1)

data=data.drop(['Gender'],axis=1)


x=data.drop('Index',axis=1)
y=data['Index']


xtrain,xtest,ytrain,ytest=train_test_split(x,y)


#classification -knn - k_Nearest_Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(xtrain,ytrain)
knn.score(xtest,ytest)


# In[10]:


def bmi():
    sam=[]
    out=['Extremely Weak','Weak','Normal','Overweight','Obesity','Extreme Obesity']
    gender=input('Are you male or female? ')
    height=int(input('Enter your Height(cm): '))
    weight=int(input('Enter your Weight(kg): '))
    sam.append(height)
    sam.append(weight)
    if gender.lower() == 'male':
        sam.append(0)
        sam.append(1)
    else:
        sam.append(1)
        sam.append(0)
    return out[knn.predict([sam])[0]]
    


# In[11]:


data=pd.read_csv('D:/data/Training.csv')
df = pd.DataFrame(data)
test_data = pd.read_csv("D:/data/Testing.csv")


# In[12]:


cols=df.columns
cols=cols[:-1]


# In[13]:


x = df[cols]
y = df['prognosis']


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[15]:


mnb = MultinomialNB()
mnb = mnb.fit(x, y)
mnb.score(x, y)
from sklearn.model_selection import cross_val_score
print ("cross result========")
scores =cross_val_score(mnb, x_test, y_test, cv=3)
print (scores)
print (scores.mean())
testx = test_data[cols]
testy = test_data['prognosis']
mnb.score(testx, testy)


# In[16]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print ("DecisionTree")

dt = DecisionTreeClassifier()

clf_dt=dt.fit(x_train,y_train)

print ("Acurracy: ", clf_dt.score(x_test,y_test))


from sklearn import metrics
from sklearn.model_selection import cross_val_score
print ("cross result========")
scores = cross_val_score(dt, x_test, y_test, cv=3)
print (scores)
print (scores.mean())


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols
# Print the feature ranking
print("Feature ranking:")
feature_dict = {}
for i,f in enumerate(features):
    feature_dict[f] = i


# In[18]:



def pre():
    sym=[]
    feat=[]
    finsym=[]
    for f in range(7):
        feat.append(features[indices[f]])
    for count in range(4):
        for f in range(7):
            if count>0 and f==0:
                print("You previously entered-->  %s " % ( finsym ))
            print("%d. Sympton -> %s " % (f+1, feat[f] ))

        s=input('Enter the index number of the symptom you are facing')
        li=s.split(' ')
        sym=[]
        for i in li:
            finsym.append(feat[int(i)-1])
            sym.append(feat[int(i)-1])
        l=[]
        k=df
        for i in sym:
            k=k[k[i]>0]  
        i=0
        while i<132:
            l.append(len(k[k[k.columns[i]]>0]))
            i+=1
        s=list(l)
        l.sort(reverse=True)
        col=[]
        feat=[]
        for i in k.columns:
            col.append(i)        
        o=0
        it=0
        while it<7:
            if col[s.index(l[o])] in finsym:
                o+=1
                continue
            feat.append(col[s.index(l[o])])
            col.remove(feat[it])
            o+=1
            it+=1
        
    return finsym


# In[19]:


def get_disease():
    sym=pre()
    symptoms=[]
    for f in sym:
        symptoms.append(feature_dict[f])
    sample_x = [1 if i in symptoms else i*0 for i in range(len(features))]
    sample_x = np.array(sample_x).reshape(1,len(sample_x))
    return dt.predict(sample_x)[0]


# In[20]:


# import our chat-bot intents file
import json
with open('D:/data/intent.json') as json_data:
    intents = json.load(json_data)


# In[21]:


words = []
classes = []
documents = []
response=[]
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
for intent in intents['intents']:
    for pattern in intent['responses']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to documents in our corpus
        response.append((w, intent['tag']))
        
# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique stemmed words", words)


# In[22]:


# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word - create base word, in attempt to represent related words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])


# In[23]:


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))


# In[24]:


# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[25]:


# Fit the model
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)


# In[27]:


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    so=0
                    #print ("found in bag: %s" % w)

    return(np.array(bag))


# In[28]:


p = bow("Load blood pessure for patient", words)
print (p)
print (classes)


# In[29]:


inputvar = pd.DataFrame([p], dtype=float, index=['input'])

x=model.predict(inputvar)


# In[30]:


def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    # generate probabilities from the model
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    # filter out predictions below a threshold, and provide intent index
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
        
    # return tuple of intent and probability
    res=[]
    for i in response:
        if classes[r[0]] in i:
            st=""
            for x in i[0]:
                st+=x+" "
            res.append(st)
    v=random.randrange(0,len(res))
    ret=[]
    ret.append(res[v])
    ret.append(classes[r[0]])
    return ret
    


# In[31]:


classify_local('Hello')


# In[36]:


def rob():
    flag=True
    print("ROBO: My name is Robo.")
    print(">>I can guide you through Symptoms of Diseases")
    print(">>I can help in predicting Disease through symptoms you tell me")
    print(">>I can also tell your Body Mass Index")
    print(">>Tell me how you are feeling and I will respond;)")
    print(">>If you want to exit, type Bye!")
    name=input("ROBO: What's your name so we can get started!")
    name=name.upper()
    while(flag==True):
        user_response = input(name+": ")
        user_response=user_response.lower()
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print("ROBO: You are welcome.. "+name)
            else:
                res=classify_local(user_response)
                st=res[1]
                if st=='find_disease':
                    print("Robo:",res[0])
                    print("Robo: "+name+" there is a chance of you having -> "+get_disease())
                    print("Robo: What else can I do for you "+name)
                elif st=='find_bmi':
                    print("Robo:",res[0])
                    print("Robo: "+name+" your Body Mass Index -> "+bmi())
                    print("ROBO: What else can I do for you "+name)
                elif st in cat:
                    print("ROBO:",res[0]+emot(st))
                elif st=='find_symptoms':
                    print("Robo:",res[0])
                    sym(find_sym())
                    print("ROBO: What else can I do for you "+name)
                else:
                    print("ROBO:",res[0])


        else:
            flag=False
            print("ROBO: Bye! take care..")


# In[37]:


rob()


# In[ ]:




