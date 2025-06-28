import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Step 1: Load the data
movies = pd.read_csv("movies.csv")

# Step 2: Preprocess genre text
movies["genres"] = movies["genres"].fillna("")

# Step 3: TF-IDF Vectorizer on genres
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# Step 4: Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 5: Build title-to-index mapping
indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Step 6: Recommendation function
def recommend(title, num_recommendations=5):
    if title not in indices:
        return f"Movie '{title}' not found in the database."

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies["title"].iloc[movie_indices].tolist()

# Example usage
user_input = input("Enter a movie you like: ")
recommendations = recommend(user_input)
print("\nRecommended Movies:")
for i, movie in enumerate(recommendations, start=1):
    print(f"{i}. {movie}")
