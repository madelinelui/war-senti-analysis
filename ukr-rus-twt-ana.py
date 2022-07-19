import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
import re 
from nltk.corpus import stopwords 
import string 

data = pd.read_csv("ukr-rus-dat.csv")
print(data.head())
print(data.columns)
data = data[["username", "tweet", "language"]]
#check for columns containing null values
data.isnull().sum()

data["language"].value_counts()
print(data["language"].value_counts())

#removing language errors, punctuation and links
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]', '', text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','', text)
    text=re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text=re.sub('\n', '', text)
    text=re.sub('\w*\d\w*', '', text)
    text=[word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"] = data["tweet"].apply(clean)

#wordcloud
text=" ".join(i for i in data.tweet)
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data["Positive"]=[sentiments.polarity_scores(i)["pos"] for i in data["tweet"]]
data["Negative"]=[sentiments.polarity_scores(i)["neg"] for i in data["tweet"]]
data["Neutral"]=[sentiments.polarity_scores(i)["neu"] for i in data["tweet"]]
data2=data[["tweet", "Positive", "Negative", "Neutral"]]
print(data.head())

positive =' '.join([i for i in data['tweet'][data['Positive']>data["Negative"]]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords, background_color='pink').generate(positive)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

negative =' '.join([i for i in data['tweet'][data['Negative']>data["Positive"]]])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords, background_color='pink').generate(negative)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()