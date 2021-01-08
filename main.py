import streamlit as st
import pandas as pd
import numpy as np

#Importing Libraries
import numpy as np
import pandas as pd

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #111;
}
</style>
    """, unsafe_allow_html=True)

@st.cache
def load_data():
    data = pd.io.parsers.read_csv('ratings.dat',
        names=['user_id', 'movie_id', 'rating', 'time'],
        engine='python', delimiter='::',encoding='latin-1')
    data = data[data['movie_id'] < 100]

    return data

@st.cache
def load_movie_data():
    movie_data = pd.io.parsers.read_csv('movies.dat',
                                        names=['movie_id', 'title', 'genre'],
                                        engine='python', delimiter='::', encoding='latin-1')
    movie_data=  movie_data[movie_data['movie_id'] < 100]

    return movie_data


st.title("Your personal movie BUDDY!")
data = load_data()
movie_data = load_movie_data()

ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    'You would definitely love these'
    i = 0;
    for id in top_indexes + 1:
        "number",i,":",movie_data[movie_data.movie_id == id].title.values[0]
        i=i+1

option = st.selectbox('Tell me, what is your favourite movie?', movie_data["title"])

movie_id = movie_data[movie_data.title == option].movie_id.values[0] 
sliced = V.T[:, :100] # representative data
indexes = top_cosine_similarity(sliced, movie_id, 3)
print_similar_movies(movie_data, movie_id, indexes+1)