import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer


def load_data():
    """Load and preprocess movie and rating datasets."""
    try:
        # Load datasets
        movies = pd.read_csv('data/title.basics.filtered.final.tsv', sep='\t', quoting=3, on_bad_lines='skip')
        ratings = pd.read_csv('data/title.ratings.filtered.tsv', sep='\t', quoting=3, on_bad_lines='skip')

        # Filter movies
        movies = movies[movies['titleType'] == 'movie']

        # Merge datasets
        return pd.merge(movies, ratings, on='tconst', how='inner')
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None


def get_genres(movies_ratings):
    """Extract unique genres from the dataset."""
    genres = set()
    for genre_list in movies_ratings['genres'].dropna():
        genres.update(genre_list.split(','))
    return sorted(genres)


def filter_movies_by_genre(movies_ratings, selected_genre):
    """Filter movies by the selected genre."""
    return movies_ratings[movies_ratings['genres'].str.contains(selected_genre, na=False)]


def get_user_ratings(filtered_movies):
    """
    Prompt the user to rate movies.
    Allows users to skip movies or stop rating.
    """
    print("\nPlease rate the following movies on a scale of 1 to 5 (1 = worst, 5 = best).")
    print("Enter 's' to skip a movie or 'q' to quit.\n")
    
    sampled_movies = filtered_movies.sample(min(10, len(filtered_movies)))  # Limit to 10 movies
    user_ratings = {}
    
    for _, row in sampled_movies.iterrows():
        while True:
            try:
                rating = input(f"Rate '{row['primaryTitle']}': ")
                if rating.lower() == 'q':  # Allow quitting
                    return user_ratings
                elif rating.lower() == 's':  # Allow skipping
                    break
                rating = int(rating)
                if 1 <= rating <= 5:
                    user_ratings[row['tconst']] = rating
                    break
                else:
                    print("Please enter a rating between 1 and 5.")
            except ValueError:
                print("Invalid input. Enter a number, 's' to skip, or 'q' to quit.")
    return user_ratings


def prepare_knn_model(ratings):
    """
    Prepare and return the k-NN model trained on movie-user matrix.
    """
    ratings_matrix = ratings.pivot(index='tconst', columns='averageRating', values='numVotes').fillna(0)
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    ratings_matrix_imputed = imputer.fit_transform(ratings_matrix)

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(ratings_matrix_imputed)

    return knn, ratings_matrix, ratings_matrix_imputed


def recommend_movies(knn, ratings_matrix, ratings_matrix_imputed, user_ratings, movies_ratings, k=5):
    """
    Recommend movies based on user ratings using the k-NN model.
    """
    recommended_movies = set()
    for movie_id, rating in user_ratings.items():
        if rating >= 4:  # Only consider movies rated 4 or 5
            if movie_id in ratings_matrix.index:
                movie_index = ratings_matrix.index.get_loc(movie_id)
                distances, indices = knn.kneighbors(
                    ratings_matrix_imputed[movie_index].reshape(1, -1), n_neighbors=k + 1
                )
                for idx in indices.flatten():
                    if idx != movie_index:  # Exclude the input movie
                        recommended_movies.add(ratings_matrix.index[idx])
    
    # Fetch titles of recommended movies
    recommendations = []
    for movie_id in recommended_movies:
        movie = movies_ratings[movies_ratings['tconst'] == movie_id]
        if not movie.empty:
            recommendations.append(movie.iloc[0]['primaryTitle'])
    return recommendations


def interactive_recommendation_system():
    """
    Main function to run the interactive movie recommendation system.
    """
    print("Welcome to the Movie Recommendation System!\n")
    
    # Load datasets
    movies_ratings = load_data()
    if movies_ratings is None:
        return
    
    # Step 1: Show genres and let the user pick one
    genres = get_genres(movies_ratings)
    print("Available genres:")
    for i, genre in enumerate(genres, start=1):
        print(f"{i}. {genre}")
    
    while True:
        try:
            genre_choice = int(input("\nSelect a genre by entering its number: "))
            if 1 <= genre_choice <= len(genres):
                selected_genre = genres[genre_choice - 1]
                break
            else:
                print("Please select a valid number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Step 2: Filter movies by the selected genre
    filtered_movies = filter_movies_by_genre(movies_ratings, selected_genre)
    if filtered_movies.empty:
        print("No movies found for the selected genre.")
        return
    
    # Step 3: Prompt user to rate random movies
    user_ratings = get_user_ratings(filtered_movies)
    if not user_ratings:
        print("No ratings provided. Exiting.")
        return
    
    # Step 4: Train k-NN model
    knn, ratings_matrix, ratings_matrix_imputed = prepare_knn_model(movies_ratings)
    
    # Step 5: Recommend movies based on user ratings
    recommendations = recommend_movies(knn, ratings_matrix, ratings_matrix_imputed, user_ratings, movies_ratings)
    if recommendations:
        print("\nWe recommend the following movies based on your ratings:")
        for movie in recommendations:
            print(f"- {movie}")
    else:
        print("\nNo recommendations could be generated. Try rating more movies.")


# Run the system
interactive_recommendation_system()
