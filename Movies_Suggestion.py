
import math
import numpy as np
import pandas as pd

# Data Reading
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge title column
movies = movies.merge(credits,on='title') 

# Remove empty data
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.isnull().sum()
movies.dropna(inplace=True)
print(movies.head())

# apply abstract syntax trees (ast) module to clean data from strings
import ast
def simple_convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L
movies['genres'] = movies['genres'].apply(simple_convert)
movies['keywords'] = movies['keywords'].apply(simple_convert)
movies['cast'] = movies['cast'].apply(simple_convert)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
pd.set_option("display.max_columns", None)
print(movies.sample(1))

# collect director name for each movie
def get_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L
movies['crew'] = movies['crew'].apply(get_director)
pd.set_option("display.max_columns", None)
print(movies.sample(1))

def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1
movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)
pd.set_option("display.max_columns", None)
print(movies.sample(1))

movies['overview'] = movies['overview'].apply(lambda x:x.split())
print(movies.sample(5))

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
pd.set_option("display.max_columns", None)
print(movies.sample(5))

df = movies.drop(columns=['overview','genres','keywords','cast','crew'])
print(df.sample(10))

df['tags'] = df['tags'].apply(lambda x: " ".join(x))
print(df['tags'])


# Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
data = cv.fit_transform(df['tags']).toarray()
# print(data)


# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(data)


def movie_recommend(movie):
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:11]:
        print(df.iloc[i[0]].title)
        # print(distances)

movie_recommend("Iron Man")


# Next We will see Colloborative Based ...














