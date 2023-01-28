
"""
Movie recommendation system

"""

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Importing the movies dataset
movies_data = pd.read_csv("C:/Users/rohu1/Desktop/Movie Recommendation System/movies.csv")


# selecting the relevant features for recommendation
selected_features = ['genres','keywords','tagline','cast','director']


# replacing the null valuess with null string
for feature in selected_features:
  movies_data[feature] = movies_data[feature].fillna('')
  
  
# combining all the 5 selected features
combined_features = movies_data['genres'] + ' ' + \
                    movies_data['keywords'] + ' ' + \
                    movies_data['tagline'] + ' ' + \
                    movies_data['cast'] + ' ' + \
                    movies_data['director']  
                    
                    
        
# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)


# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)

# getting the movie name from the user
movie_name = input('PLEASE ENTER YOUR FAVORITE MOVIE NAME HERE--')
                    
                    
                    
# creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
                    
                    
# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
close_match = find_close_match[0]
                    
# finding the index of the movie with title
index_of_the_movie = movies_data.loc[(movies_data.title == close_match)]['index'].tolist()[0]


# getting a similarity score for all the movie against the movie entered by the user
similarity_score = list(similarity[index_of_the_movie])


# Create a dataframe with similarity scores and correspoding movie names for the movie entered by the user
similarity_score_df = pd.DataFrame()
similarity_score_df ['similarity_score'] = similarity_score
similarity_score_df ['Title'] = movies_data['title']
similarity_score_df ['Original_index'] = movies_data['index']

#Sorting based on similarity_score
similarity_score_df = similarity_score_df.sort_values(by=['similarity_score'], ascending=False)


#Take the top 20 and present to the user
top_picks = similarity_score_df ['Title'].tolist()[:20]
print(top_picks)









                    
                    
                    