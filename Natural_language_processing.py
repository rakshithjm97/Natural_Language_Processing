# Natural Language Processing


# =============================================================================
# Importing the librarie
# =============================================================================
import os # working directory
import matplotlib.pyplot as plt #  plots
import pandas as pd # data handling
import re # regex --> pattern matching
import nltk # NLP Library
from nltk.corpus import stopwords # importing stopwords to remove
from nltk.stem.porter import PorterStemmer # stemming
from nltk.stem import WordNetLemmatizer # Lemmitization
import string # regex
from wordcloud import WordCloud # Generating word clouds
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # Sentiment Analysis
nltk.download('stopwords') # Downwloding the stopwords
nltk.download('wordnet') # Downwloding the word 
nltk.download('omw-1.4') # Downwloding the omw

# =============================================================================
# User Inputs
# =============================================================================
# Importing the dataset
os.chdir(R'F:\data_science\nlp\Restaurant_Reviews.tsv')
os.getcwd()
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
dataset.shape

# =============================================================================
# 1st Block: Analysis of Restaruant Review
# =============================================================================

print("The number of Positive reviews=", (dataset["Liked"]==1).sum())
print("The number of Negative reviews=", (dataset["Liked"]==0).sum())


#Defining empty text of Postitve Reviews and Negative Reviews
pos_rev=" "
neg_rev=" "

#1. Creation of Positive & Negative Review Text

"""Appending all the Postive Review and Negative Reviews Seperately"""

for i in range(0,dataset.shape[0]):
    if dataset["Liked"][i]==1:
        pos_rev=pos_rev+" "+dataset["Review"][i] # appending +ve reviews
    if dataset["Liked"][i]==0:
        neg_rev=neg_rev+" "+dataset["Review"][i] # appending -ve reviews


print("Length of positive reviews-",len(pos_rev))
print("")
print("Part of positive reviews-")
print(pos_rev[1000:2000])


print("Length of negative reviews-",len(neg_rev))
print("")
print("Part of negative reviews-")
print(neg_rev[1000:2000])

#2. Cleaning of Text by removing the Punctuation Marks
""" removal of punctuations """
string.punctuation

#Creating empty with Punctuation removal
text_nopunct_pos=''#initating the variable with an empty string
text_nopunct_pos= "".join([char for char in pos_rev if char not in string.punctuation])


#3. Creating the Tokens from the Aggregated String
""" Creating Tokens from the Text """

#Initating the tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')

#Tokenizing the text
pos_tokens = tokenizer.tokenize(text_nopunct_pos)
len(pos_tokens)

#4. Normalizing the Text by converting them into lowercase
""" Lowercase conversion """

#now we shall make everything lowercase for uniformity
#to hold the new lower case words

words_pos = [] #Initating empty list 

# Looping through the tokens and make them lower case
for word in pos_tokens:
    words_pos.append(word.lower())

    
#5. Removing the Stopwords from the Tokens
""" Stopword Removals """

#Stop words are generally the most common words in a language.
#English stop words from nltk.    
final_words_pos=[]#Empty List for appending without stopwords

#Now we need to remove the stop words from the words variable
#Appending to words_new all words that are in words but not in sw
stop_words = set(stopwords.words("english"))

for word in words_pos:
    if word not in stop_words:
        final_words_pos.append(word)   
        
print("Final List of Postive Tokens is complete!")

#6. Extracting the Lemmatized words
""" Lemmatization """
    
wn = WordNetLemmatizer() #Initating Lemmatization token, returns the root/base words which returns meaningful words

lem_words_pos=[]#Empty List to contain lematized words

for word in final_words_pos:
    word=wn.lemmatize(word)#Lemmitization
    lem_words_pos.append(word)
    
 
#7. Extracting the Stemmed words
""" Stemming """
    
ps = PorterStemmer() #Initating Stemming token, returns the root/base words 

stem_words_pos=[]#Empty List to contain lematized words

for word in final_words_pos:
    word=ps.stem(word)# Stemming
    stem_words_pos.append(word)
    
#8. Create the Frequency Distribution of the Lemmatized Words
""" Frequency Distribution of Postive/Negative Tokens """
      
#The frequency distribution of the words
freq_dist_pos = nltk.FreqDist(lem_words_pos)
freq_dist_pos_stem = nltk.FreqDist(stem_words_pos)


#Frequency Distribution Plot
plt.subplots(figsize=(20,12))
freq_dist_pos.plot(30)

#9. Create the Word Clouds based on tokens
""" Creating Word Clouds """

#converting into string
res_pos=' '.join([i for i in lem_words_pos if not i.isdigit()])

plt.subplots(figsize=(16,10))
wordcloud = WordCloud(
                          background_color='black',
                          max_words=100,
                          width=1400,
                          height=1200
                         ).generate(res_pos)


plt.imshow(wordcloud)
plt.title('Positive Reviews World Cloud (100 words)')
plt.axis('off')
plt.show()



# =============================================================================
# 2nd Block: Modeling on the Reviews
# =============================================================================

corpus = []#Definining the corpus matrix

for i in range(0, dataset.shape[0]):
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    
    # convert all cases to lower cases 
    review = review.lower()
        
    # split to array(default delimiter is " ") 
    review = review.split()
    
    # creating PorterStemmer object to take main stem of each word 
    #ps = PorterStemmer()
    
    # loop for stemming each word in string array at ith row 
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    #Applying Lemmatization
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stop_words)]
    
    
    # rejoin all string array elements to create back into a string 
    review = ' '.join(review)
    corpus.append(review)


# =============================================================================
# Creating the Bag of Words model 
# =============================================================================

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import classification_report 
print(classification_report(y_pred, 
            y_test))
    


# =============================================================================
# Trying other predictive models
# =============================================================================

# Fitting Decision Trees to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier_DT.fit(X_train, y_train)


# Predicting the Test set results
y_pred_DT = classifier_DT.predict(X_test)

cm_DT = confusion_matrix(y_test, y_pred_DT)
print(classification_report(y_pred_DT, 
            y_test))


# Fitting Random Forest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=1000, criterion = 'entropy',random_state = 0)

classifier_rf.fit(X_train, y_train)


# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
print(classification_report(y_pred_rf, 
            y_test))



# =============================================================================
# Building the Model with TF-IDF Transformer
# =============================================================================

X = dataset.iloc[:,0].values
y = dataset.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

from sklearn.feature_extraction.text import TfidfTransformer
#create pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', RandomForestClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


pipeline.fit(X_train, y_train)

#using pipeline to predict
predictions = pipeline.predict(X_test)

#printing confusion matrix and classification report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


