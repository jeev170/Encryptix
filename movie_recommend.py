import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class TamilMovieRecommender:
    def __init__(self):
        """Initialize the Tamil movie recommender system with sample data."""
        # Sample user ratings for movies (rows=users, columns=movies)
        # Rating scale: 1-5, where 0 means not rated
        self.ratings = np.array([
            [5, 3, 0, 1, 4, 0, 3, 0, 5, 0, 4, 0, 3, 0, 2, 5, 0, 4, 0, 3],  # User 1
            [4, 0, 0, 1, 0, 5, 0, 4, 0, 2, 0, 5, 0, 3, 0, 0, 4, 0, 2, 0],  # User 2
            [0, 5, 4, 0, 3, 0, 0, 0, 2, 0, 5, 0, 4, 0, 3, 0, 0, 5, 0, 1],  # User 3
            [2, 0, 5, 0, 0, 1, 0, 4, 0, 3, 0, 4, 0, 0, 5, 0, 3, 0, 0, 4],  # User 4
            [0, 4, 0, 3, 5, 0, 2, 0, 0, 0, 3, 0, 5, 0, 0, 4, 0, 3, 5, 0],  # User 5
            [0, 0, 0, 4, 0, 0, 5, 0, 3, 1, 0, 0, 3, 5, 0, 0, 4, 0, 2, 0],  # User 6
            [5, 3, 0, 0, 0, 0, 0, 2, 0, 4, 0, 5, 0, 0, 3, 0, 0, 4, 0, 5],  # User 7
            [0, 0, 4, 0, 3, 5, 0, 0, 1, 0, 3, 0, 0, 4, 0, 5, 0, 0, 3, 0],  # User 8
            [3, 0, 0, 5, 0, 0, 4, 0, 0, 2, 0, 3, 0, 0, 4, 0, 5, 0, 0, 3],  # User 9
            [0, 2, 0, 0, 4, 0, 0, 3, 0, 5, 0, 0, 5, 0, 0, 3, 0, 4, 0, 0]   # User 10
        ])
        
        # Tamil Movie titles
        self.movies = [
            "Baahubali: The Beginning", "Vikram", "Master", "Ponniyin Selvan: I", "Jailer",
            "Vada Chennai", "Super Deluxe", "96", "Kaithi", "Asuran",
            "Leo", "Karnan", "Soorarai Pottru", "Pariyerum Perumal", "Jai Bhim",
            "Peranbu", "Ratsasan", "Mersal", "K.G.F: Chapter 1", "Viswasam"
        ]
        
        # Movie genres (action, drama, thriller, etc.)
        self.genres = [
            ["Action", "Historical", "Epic"], ["Action", "Thriller", "Crime"],
            ["Action", "Thriller", "Drama"], ["Historical", "Epic", "Drama"],
            ["Action", "Comedy", "Drama"], ["Crime", "Action", "Drama"],
            ["Drama", "Comedy", "Anthology"], ["Romance", "Drama"],
            ["Action", "Thriller"], ["Action", "Drama"],
            ["Action", "Crime", "Thriller"], ["Action", "Drama"],
            ["Drama", "Biography"], ["Drama", "Social"],
            ["Legal", "Drama", "Crime"], ["Drama", "Family"],
            ["Thriller", "Crime", "Mystery"], ["Action", "Thriller"],
            ["Action", "Drama", "Thriller"], ["Action", "Drama", "Family"]
        ]
        
        # Directors, years, and actors
        self.directors = [
            "S.S. Rajamouli", "Lokesh Kanagaraj", "Lokesh Kanagaraj", "Mani Ratnam", 
            "Nelson", "Vetrimaaran", "Thiagarajan Kumararaja", "C. Prem Kumar", 
            "Lokesh Kanagaraj", "Vetrimaaran", "Lokesh Kanagaraj", "Mari Selvaraj", 
            "Sudha Kongara", "Mari Selvaraj", "T.J. Gnanavel", "Ram", 
            "Ram Kumar", "Atlee", "Prashanth Neel", "Siva"
        ]
        
        # Compute user similarity matrix
        self.user_similarity = self._compute_user_similarity()
        
    def _compute_user_similarity(self):
        """Compute similarity between users based on their ratings."""
        # Create masked array to ignore missing ratings
        ratings_matrix = np.ma.masked_values(self.ratings, 0)
        
        # Compute cosine similarity and set self-similarity to 0
        similarity = cosine_similarity(ratings_matrix)
        np.fill_diagonal(similarity, 0)
        
        return similarity
    
    def recommend_movies(self, user_id, n_recommendations=3):
        """Recommend movies using collaborative filtering."""
        # Get user's ratings and unrated movies
        user_ratings = self.ratings[user_id]
        unrated_movies = np.where(user_ratings == 0)[0]
        
        if len(unrated_movies) == 0:
            return "You have rated all available movies!"
        
        # Get similar users (sorted by similarity)
        similar_users = np.argsort(self.user_similarity[user_id])[::-1]
        
        # Calculate predicted ratings for unrated movies
        predicted_ratings = {}
        for movie_id in unrated_movies:
            movie_ratings = self.ratings[:, movie_id]
            
            # Skip if no one has rated this movie
            if np.sum(movie_ratings > 0) == 0:
                continue
            
            weighted_sum = similarity_sum = 0
            
            # Calculate weighted average rating
            for other_user in similar_users:
                if self.ratings[other_user, movie_id] == 0:
                    continue
                
                similarity = self.user_similarity[user_id, other_user]
                weighted_sum += similarity * self.ratings[other_user, movie_id]
                similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_ratings[movie_id] = weighted_sum / similarity_sum
        
        # Sort and return top recommendations
        sorted_movies = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        recommendations = [(self.movies[m_id], round(score, 2)) for m_id, score in sorted_movies[:n_recommendations]]
        
        return recommendations
    
    def get_user_rated_movies(self, user_id):
        """Get movies rated by the specified user."""
        rated_movies = []
        for movie_id, rating in enumerate(self.ratings[user_id]):
            if rating > 0:
                rated_movies.append((self.movies[movie_id], rating))
        return rated_movies
    
    def add_rating(self, user_id, movie_id, rating):
        """Add or update a movie rating."""
        if 0 <= user_id < len(self.ratings) and 0 <= movie_id < len(self.movies):
            self.ratings[user_id, movie_id] = rating
            self.user_similarity = self._compute_user_similarity()
            return True
        return False
    
    def get_movie_id(self, movie_title):
        """Get movie ID by title."""
        try:
            return self.movies.index(movie_title)
        except ValueError:
            return -1
    
    def get_movie_average_rating(self, movie_id):
        """Get the average rating for a movie."""
        ratings = self.ratings[:, movie_id]
        rated = ratings > 0
        return np.mean(ratings[rated]) if np.sum(rated) > 0 else 0
    
    def get_most_popular_movies(self, n=3):
        """Get the most popular movies by average rating."""
        avg_ratings = [(i, self.get_movie_average_rating(i)) 
                      for i in range(len(self.movies))]
        
        # Filter out unrated movies, sort by rating
        rated_movies = [(i, r) for i, r in avg_ratings if r > 0]
        sorted_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)
        
        # Return top n popular movies
        return [(self.movies[m_id], round(rating, 2)) 
                for m_id, rating in sorted_movies[:n]]
    
    def content_based_recommendations(self, user_id, n=3):
        """Generate content-based recommendations based on genre preferences."""
        # Get user's rated movies
        rated_movies = [(m_id, rating) for m_id, rating in enumerate(self.ratings[user_id]) 
                        if rating > 0]
        
        if not rated_movies:
            return "You need to rate some movies first!"
        
        # Count genre preferences weighted by ratings
        genre_counts = {}
        for movie_id, rating in rated_movies:
            for genre in self.genres[movie_id]:
                genre_counts[genre] = genre_counts.get(genre, 0) + rating
        
        # Get top 3 favorite genres
        top_genres = [genre for genre, _ in sorted(
            genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        # Score unrated movies by genre match
        unrated_movies = np.where(self.ratings[user_id] == 0)[0]
        movie_scores = []
        for movie_id in unrated_movies:
            score = sum(1 for genre in self.genres[movie_id] if genre in top_genres)
            if score > 0:
                movie_scores.append((movie_id, score))
        
        # Return top recommendations
        sorted_recs = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        return [(self.movies[m_id], self.genres[m_id]) 
                for m_id, _ in sorted_recs[:n]]

# Example usage
def main():
    recommender = TamilMovieRecommender()
    
    while True:
        print("\n===== Tamil Movie Recommendation System =====")
        print("1. View your rated movies")
        print("2. Rate a movie")
        print("3. Get movie recommendations (collaborative filtering)")
        print("4. Get recommendations based on your genre preferences")
        print("5. View most popular movies")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            user_id = int(input("Enter your user ID (0-9): "))
            rated_movies = recommender.get_user_rated_movies(user_id)
            if rated_movies:
                print(f"\nMovies rated by User {user_id}:")
                for movie, rating in rated_movies:
                    print(f"- {movie}: {rating}/5")
            else:
                print(f"User {user_id} hasn't rated any movies yet.")
                
        elif choice == '2':
            user_id = int(input("Enter your user ID (0-9): "))
            
            # Display all movies
            print("\nAvailable movies:")
            for i, movie in enumerate(recommender.movies):
                print(f"{i}: {movie}")
                
            movie_id = int(input("\nEnter movie ID to rate: "))
            rating = float(input("Enter rating (1-5): "))
            
            if recommender.add_rating(user_id, movie_id, rating):
                print("Rating added successfully!")
            else:
                print("Invalid user ID or movie ID.")
                
        elif choice == '3':
            user_id = int(input("Enter your user ID (0-9): "))
            n = int(input("How many recommendations? "))
            
            recommendations = recommender.recommend_movies(user_id, n)
            if isinstance(recommendations, str):
                print(recommendations)
            else:
                print(f"\nRecommended movies for User {user_id}:")
                for movie, score in recommendations:
                    print(f"- {movie} (predicted rating: {score}/5)")
                    
        elif choice == '4':
            user_id = int(input("Enter your user ID (0-9): "))
            n = int(input("How many recommendations? "))
            
            recommendations = recommender.content_based_recommendations(user_id, n)
            if isinstance(recommendations, str):
                print(recommendations)
            else:
                print(f"\nBased on your genre preferences, you might enjoy:")
                for movie, genres in recommendations:
                    print(f"- {movie} (Genres: {', '.join(genres)})")
                    
        elif choice == '5':
            n = int(input("How many popular movies to show? "))
            popular_movies = recommender.get_most_popular_movies(n)
            
            print("\nMost popular movies:")
            for movie, rating in popular_movies:
                print(f"- {movie} (Average rating: {rating}/5)")
                
        elif choice == '6':
            print("Thank you for using the Tamil Movie Recommendation System!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
