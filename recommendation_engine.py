import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path='movies_preprocessed_model.csv', use_cache=True):
        """Initialize the Movie Recommender with data loading and preprocessing"""
        # Use sample datasets if full datasets don't exist (for deployment)
        import os
        
        # Check if full dataset exists, otherwise use sample
        if os.path.exists(data_path):
            full_data_path = 'movies_cleaned_full.csv'
            print(f'🎬 Loading {data_path} ...')
        elif os.path.exists('movies_preprocessed_model_sample.csv'):
            data_path = 'movies_preprocessed_model_sample.csv'
            full_data_path = 'movies_cleaned_full_sample.csv'
            print("📦 Using sample dataset (20K movies) for deployment")
            print(f'🎬 Loading {data_path} ...')
        else:
            raise FileNotFoundError("No dataset files found! Need either full or sample CSV files.")
        
        self.model_df = pd.read_csv(data_path, low_memory=False)
        
        # Load full cleaned CSV for poster paths and extra fields
        print(f'📂 Loading full dataset for poster paths...')
        self.full_df = pd.read_csv(full_data_path, low_memory=False)
        
        required_cols = ['title', 'genres', 'keywords', 'cast', 'directors', 'overview']
        missing = [c for c in required_cols if c not in self.model_df.columns]
        if missing:
            raise ValueError(f'Missing columns in dataset: {missing}')
        
        self.model_df = self.model_df[self.model_df['title'].notna()].reset_index(drop=True)
        
        # Filter movies with rating >= 6.0
        if 'vote_average' in self.model_df.columns:
            initial_count = len(self.model_df)
            self.model_df = self.model_df[self.model_df['vote_average'] >= 6.0].reset_index(drop=True)
            print(f'🎯 Filtered {initial_count - len(self.model_df):,} movies below 6.0 rating')
        
        if 'vote_average' in self.full_df.columns:
            self.full_df = self.full_df[self.full_df['vote_average'] >= 6.0].reset_index(drop=True)
        
        # Parse list columns
        for col in ['genres', 'keywords', 'cast', 'directors']:
            self.model_df[col] = self.model_df[col].apply(self._parse_list_col)
            if col in self.full_df.columns:
                self.full_df[col] = self.full_df[col].apply(self._parse_list_col)
        
        self.model_df['overview'] = self.model_df['overview'].fillna('')
        
        print(f'📊 Dataset size: {len(self.model_df):,} movies')
        print(f'🖼️ Full dataset size (with posters): {len(self.full_df):,} movies')
        
        # Define mood-based filters
        self._define_moods()
        
        # Build features and TF-IDF matrix
        self._build_features()
    
    def _define_moods(self):
        """Define mood categories based on genres and keywords"""
        self.moods = {
            "Cozy": {
                "genres": ["Romance", "Comedy", "Family", "Animation"],
                "keywords": ["friendship", "love", "holiday", "christmas", "family", "heartwarming", "feel good"],
                "exclude_genres": ["Horror", "Thriller"]
            },
            "Heart-breaking": {
                "genres": ["Drama", "Romance", "War"],
                "keywords": ["death", "tragedy", "loss", "sad", "emotional", "terminal illness", "sacrifice"],
                "min_rating": 7.0
            },
            "Mind-bending": {
                "genres": ["Science Fiction", "Thriller", "Mystery"],
                "keywords": ["time travel", "parallel universe", "psychological", "twist", "mind", "reality", "dream", "conspiracy"],
                "min_rating": 6.5
            },
            "Feel-good": {
                "genres": ["Comedy", "Romance", "Family", "Music"],
                "keywords": ["happy", "uplifting", "inspiring", "heartwarming", "triumph", "success", "joy"],
                "exclude_genres": ["Horror", "Thriller", "War"],
                "min_rating": 6.0
            },
            "Dark": {
                "genres": ["Horror", "Thriller", "Crime", "Mystery"],
                "keywords": ["murder", "violence", "serial killer", "revenge", "corruption", "noir", "disturbing", "psychological"],
                "exclude_keywords": ["comedy"]
            },
            "Slow & Beautiful": {
                "genres": ["Drama", "Romance"],
                "keywords": ["art", "cinematography", "slow burn", "contemplative", "poetic", "visual", "atmospheric", "quiet"],
                "min_rating": 7.0,
                "exclude_genres": ["Action", "Horror"]
            },
            "Epic Adventure": {
                "genres": ["Adventure", "Fantasy", "Action"],
                "keywords": ["quest", "journey", "epic", "hero", "sword", "battle", "adventure", "exploration", "treasure"],
                "min_rating": 6.5
            },
            "Nostalgic": {
                "genres": ["Drama", "Comedy", "Romance"],
                "keywords": ["nostalgia", "childhood", "memory", "past", "80s", "90s", "coming of age", "retro", "vintage"],
                "min_rating": 6.0
            },
            "Intense & Gripping": {
                "genres": ["Thriller", "Action", "Crime", "War"],
                "keywords": ["suspense", "tension", "edge of seat", "intense", "survival", "chase", "hostage", "escape"],
                "min_rating": 6.5
            },
            "Quirky & Weird": {
                "genres": ["Comedy", "Fantasy", "Science Fiction"],
                "keywords": ["strange", "weird", "surreal", "absurd", "bizarre", "offbeat", "quirky", "unconventional", "cult"],
                "min_rating": 5.5
            },
            "Inspiring": {
                "genres": ["Drama", "Biography", "Sport"],
                "keywords": ["true story", "inspiring", "overcome", "determination", "triumph", "courage", "perseverance", "achievement"],
                "min_rating": 6.5
            },
            "Scary": {
                "genres": ["Horror", "Thriller"],
                "keywords": ["scary", "terrifying", "fear", "haunted", "ghost", "demon", "monster", "nightmare", "supernatural"],
                "exclude_genres": ["Comedy"]
            }
        }
        
    def _parse_list_col(self, x):
        """Parse string representation of list into actual list"""
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        if isinstance(x, str) and x == "":
            return []
        try:
            parsed = ast.literal_eval(str(x))
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed]
        except Exception:
            pass
        return [i.strip() for i in str(x).split(",")]
    
    def _clean_string(self, text):
        """Remove spaces from multi-word terms"""
        return str(text).replace(" ", "").replace("-", "")
    
    def _create_soup(self, row):
        """
        Build weighted feature string for each movie
        🏆 OPTIMAL WEIGHTS (from Jupyter notebook testing):
        - Genres: 7x
        - Directors: 7x
        - Cast: 2x
        - Keywords: 3x
        - Overview: 1x
        """
        genres = [self._clean_string(g) for g in row["genres"]]
        directors = [self._clean_string(d) for d in row["directors"]]
        cast = [self._clean_string(c) for c in row["cast"][:5]]
        keywords = [self._clean_string(k) for k in row["keywords"][:10]]
        overview_words = self._clean_string(row["overview"]).split()
        
        parts = []
        parts.extend(genres * 7)         # 7x weight
        parts.extend(directors * 7)      # 7x weight
        parts.extend(cast * 2)           # 2x weight
        parts.extend(keywords * 3)       # 3x weight
        parts.extend(overview_words * 1) # 1x weight
        
        return " ".join(parts)
    
    def _build_features(self):
        """Build TF-IDF matrix and cosine similarity matrix with caching"""
        print("🔧 Building feature vectors...")
        
        self.model_df["soup"] = self.model_df.apply(self._create_soup, axis=1)
        self.model_df = self.model_df[self.model_df["soup"].str.strip() != ""].reset_index(drop=True)
        
        print(f"✨ After removing empty features: {len(self.model_df):,} movies")
        
        # Check cache
        cache_dir = Path("recommendation_cache")
        tfidf_cache_file = cache_dir / "features.npz"
        
        # Try to load TF-IDF from cache (we'll compute similarities on-demand)
        cache_loaded = False
        if tfidf_cache_file.exists():
            try:
                print("📦 Loading cached TF-IDF matrix...")
                self.tfidf_matrix = sparse.load_npz(tfidf_cache_file)
                
                # Rebuild vectorizer (needed for transformations)
                self.tfidf = TfidfVectorizer(
                    stop_words="english",
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2
                )
                self.tfidf.fit(self.model_df["soup"])
                print(f"✅ Loaded TF-IDF matrix shape: {self.tfidf_matrix.shape}")
                cache_loaded = True
            except Exception as e:
                print(f"⚠️  Cache corrupted ({e}). Rebuilding from scratch...")
                cache_loaded = False
        
        if not cache_loaded:
            print("🔨 Building TF-IDF matrix from scratch...")
            self.tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                min_df=2
            )
            
            self.tfidf_matrix = self.tfidf.fit_transform(self.model_df["soup"])
            print(f"✅ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            
            # Save to cache
            cache_dir.mkdir(exist_ok=True)
            sparse.save_npz(tfidf_cache_file, self.tfidf_matrix)
            print(f"💾 Cached TF-IDF matrix to {tfidf_cache_file}")
        
        # Note: We compute cosine similarities on-demand to save memory
        
        # Create title to index mapping
        self.title_to_idx = pd.Series(self.model_df.index, index=self.model_df["title"].str.lower()).to_dict()
    
    def search_movies_advanced(self, query):
        """
        Search for movies by title, actor, or director with smart matching
        Features:
        - Case-insensitive partial matching
        - Word-by-word matching (e.g., "dark knight" matches "The Dark Knight")
        - Removes common words like "the", "a", "an"
        - Searches in title, cast, directors, and overview
        """
        if not query or len(query) == 0:
            return []
        
        query_lower = query.lower().strip()
        
        # Split query into words and remove common articles
        stop_words = {'the', 'a', 'an', 'of', 'and', 'in', 'to'}
        query_words = [w for w in query_lower.split() if w not in stop_words]
        
        # Store results with scores for ranking
        results_scored = []
        
        for _, row in self.model_df.iterrows():
            score = 0
            title_lower = row["title"].lower()
            
            # Exact substring match in title (highest priority)
            if query_lower in title_lower:
                score += 100
                results_scored.append((row["title"], score))
                continue
            
            # Word-by-word match in title
            title_words = [w for w in title_lower.split() if w not in stop_words]
            matching_words = sum(1 for qw in query_words if any(qw in tw for tw in title_words))
            if matching_words > 0:
                score += matching_words * 50  # 50 points per matching word
            
            # Check if all query words appear in title (even if not consecutive)
            if all(any(qw in tw for tw in title_words) for qw in query_words):
                score += 30
            
            # Search in actors (medium priority)
            if isinstance(row["cast"], list):
                for actor in row["cast"][:10]:  # Check top 10 actors
                    actor_lower = actor.lower()
                    if query_lower in actor_lower:
                        score += 40
                        break
                    # Word match in actor name
                    if any(qw in actor_lower for qw in query_words):
                        score += 20
            
            # Search in directors (medium-high priority)
            if isinstance(row["directors"], list):
                for director in row["directors"]:
                    director_lower = director.lower()
                    if query_lower in director_lower:
                        score += 60
                        break
                    # Word match in director name
                    if any(qw in director_lower for qw in query_words):
                        score += 30
            
            # Search in overview (low priority)
            if isinstance(row.get("overview"), str):
                overview_lower = row["overview"].lower()
                if query_lower in overview_lower:
                    score += 10
            
            # Only include movies with non-zero score
            if score > 0:
                results_scored.append((row["title"], score))
        
        # Sort by score (highest first) and return titles
        results_scored.sort(key=lambda x: x[1], reverse=True)
        return [title for title, score in results_scored]
    
    def filter_by_mood(self, mood, limit=50, min_rating=0):
        """Filter movies based on mood criteria
        
        Args:
            mood: Mood category (Cozy, Heart-breaking, Mind-bending, Feel-good, Dark, Slow & Beautiful)
            limit: Maximum number of movies to return
            min_rating: Minimum rating filter
        
        Returns:
            DataFrame of movies matching the mood criteria with poster paths
        """
        if mood not in self.moods:
            return pd.DataFrame()
        
        if self.full_df is None:
            return pd.DataFrame()
        
        mood_criteria = self.moods[mood]
        filtered_titles = []
        
        for _, row in self.model_df.iterrows():
            # Check minimum rating
            mood_min_rating = mood_criteria.get("min_rating", min_rating)
            if row.get("vote_average", 0) < mood_min_rating:
                continue
            
            # Check excluded genres
            excluded_genres = mood_criteria.get("exclude_genres", [])
            if isinstance(row.get("genres"), list):
                if any(genre in row["genres"] for genre in excluded_genres):
                    continue
            
            # Check if movie matches mood genres
            required_genres = mood_criteria.get("genres", [])
            movie_genres = row.get("genres", [])
            if isinstance(movie_genres, list) and required_genres:
                if any(genre in movie_genres for genre in required_genres):
                    filtered_titles.append(row["title"])
                    continue
            
            # Check if movie matches mood keywords
            required_keywords = mood_criteria.get("keywords", [])
            movie_keywords = row.get("keywords", [])
            if isinstance(movie_keywords, list) and required_keywords:
                if any(keyword in movie_keywords for keyword in required_keywords):
                    filtered_titles.append(row["title"])
        
        # Get movies from filtered titles and merge with poster data
        filtered_df = self.model_df[self.model_df['title'].isin(filtered_titles)]
        
        # Prepare columns to merge - only use existing columns
        merge_cols = ['title']
        if 'poster_path' in self.full_df.columns:
            merge_cols.append('poster_path')
        if 'popularity' in self.full_df.columns:
            merge_cols.append('popularity')
        if 'backdrop_path' in self.full_df.columns:
            merge_cols.append('backdrop_path')
        
        # Merge with full_df to get poster paths
        merged = filtered_df.merge(
            self.full_df[merge_cols], 
            on='title', 
            how='left',
            suffixes=('', '_full')
        )
        
        # Remove duplicates - keep the entry with highest popularity
        if 'popularity' in merged.columns:
            merged = merged.sort_values('popularity', ascending=False)
            merged = merged.drop_duplicates(subset=['title'], keep='first')
        else:
            merged = merged.drop_duplicates(subset=['title'], keep='first')
        
        # Remove movies without posters (only if poster_path column exists)
        if 'poster_path' in merged.columns:
            merged = merged[merged['poster_path'].notna()]
            merged = merged[merged['poster_path'] != '']
        else:
            # If no poster_path, add empty column as fallback
            merged['poster_path'] = ''
        
        # Sort by popularity and limit results
        if 'popularity' in merged.columns:
            merged = merged.sort_values('popularity', ascending=False)
        return merged.head(limit)
    
    def get_recommendations(self, title, top_n=10, year_tolerance=15):
        '''Get movie recommendations based on a single title'''
        title_lower = title.lower()
        if title_lower not in self.title_to_idx:
            available = list(self.title_to_idx.keys())
            matches = [t for t in available if title_lower in t]
            if matches:
                print(f'Did you mean one of these? {matches[:5]}')
            raise ValueError(f"Movie '{title}' not found in dataset")
        
        idx = self.title_to_idx[title_lower]
        
        movie_vector = self.tfidf_matrix[idx]
        cosine_similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()
        
        similar_indices = cosine_similarities.argsort()[::-1]
        
        if year_tolerance and 'release_date' in self.model_df.columns:
            movie_year = pd.to_datetime(self.model_df.iloc[idx]['release_date'], errors='coerce')
            if pd.notna(movie_year):
                movie_year = movie_year.year
                self.model_df['year'] = pd.to_datetime(self.model_df['release_date'], errors='coerce').dt.year
                
                filtered_indices = []
                for i in similar_indices:
                    if i == idx:
                        continue
                    candidate_year = self.model_df.iloc[i]['year']
                    if pd.notna(candidate_year):
                        if abs(candidate_year - movie_year) <= year_tolerance:
                            filtered_indices.append(i)
                    else:
                        filtered_indices.append(i)
                    
                    if len(filtered_indices) >= top_n:
                        break
                similar_indices = filtered_indices
            else:
                similar_indices = [i for i in similar_indices if i != idx][:top_n]
        else:
            similar_indices = [i for i in similar_indices if i != idx][:top_n]
        
        results = self.model_df.iloc[similar_indices][['title', 'vote_average', 'genres', 'directors']].copy()
        results['similarity_score'] = [cosine_similarities[i] for i in similar_indices]
        results['match_%'] = (results['similarity_score'] * 100).round(1)
        
        results['genres'] = results['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        results['directors'] = results['directors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        return results[['title', 'match_%', 'vote_average', 'genres', 'directors']].reset_index(drop=True)
    
    def get_recommendations_from_favorites(self, titles, top_n=10):
        '''Get movie recommendations based on multiple favorite titles'''
        if not isinstance(titles, list):
            titles = [titles]
        
        indices = []
        for title in titles:
            title_lower = title.lower()
            if title_lower not in self.title_to_idx:
                print(f"Warning: '{title}' not found, skipping...")
                continue
            indices.append(self.title_to_idx[title_lower])
        
        if not indices:
            raise ValueError('None of the provided titles were found')
        
        print(f'Using {len(indices)} movies as input: {titles[:len(indices)]}')
        
        favorite_vectors = self.tfidf_matrix[indices]
        avg_vector = np.asarray(favorite_vectors.mean(axis=0))
        
        cosine_similarities = cosine_similarity(avg_vector, self.tfidf_matrix).flatten()
        
        similar_indices = cosine_similarities.argsort()[::-1]
        similar_indices = [i for i in similar_indices if i not in indices][:top_n]
        
        results = self.model_df.iloc[similar_indices][['title', 'vote_average', 'genres', 'directors']].copy()
        results['similarity_score'] = [cosine_similarities[i] for i in similar_indices]
        results['match_%'] = (results['similarity_score'] * 100).round(1)
        
        results['genres'] = results['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        results['directors'] = results['directors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        return results[['title', 'match_%', 'vote_average', 'genres', 'directors']].reset_index(drop=True)
    
    def get_movie_details(self, title):
        '''Get details for a specific movie'''
        title_lower = title.lower()
        if title_lower not in self.title_to_idx:
            return None
        
        idx = self.title_to_idx[title_lower]
        movie = self.model_df.iloc[idx]
        
        return {
            'title': movie['title'],
            'genres': ', '.join(movie['genres']) if isinstance(movie['genres'], list) else str(movie['genres']),
            'directors': ', '.join(movie['directors']) if isinstance(movie['directors'], list) else str(movie['directors']),
            'cast': ', '.join(movie['cast'][:5]) if isinstance(movie['cast'], list) else str(movie['cast']),
            'overview': movie['overview'],
            'vote_average': movie.get('vote_average', 0)
        }
    
    def get_movies_with_posters(self, limit=100, min_rating=0, sort_by="popularity", search_query=None):
        """Get movies with poster paths for display"""
        if self.full_df is None:
            return pd.DataFrame()
        
        # Prepare columns to merge - only use existing columns
        merge_cols = ['title']
        if 'poster_path' in self.full_df.columns:
            merge_cols.append('poster_path')
        if 'popularity' in self.full_df.columns:
            merge_cols.append('popularity')
        if 'backdrop_path' in self.full_df.columns:
            merge_cols.append('backdrop_path')
        
        # Merge model_df with full_df to get poster paths
        merged = self.model_df.merge(
            self.full_df[merge_cols], 
            on='title', 
            how='left',
            suffixes=('', '_full')
        )
        
        # Remove duplicates - keep the entry with highest popularity for each title
        # This handles cases where multiple movies share the same title
        if 'popularity' in merged.columns:
            merged = merged.sort_values('popularity', ascending=False)
            merged = merged.drop_duplicates(subset=['title'], keep='first')
        else:
            # If no popularity column, keep first occurrence
            merged = merged.drop_duplicates(subset=['title'], keep='first')
        
        # Apply search filter FIRST (before limiting)
        if search_query and len(search_query) > 0:
            # Use advanced search to find by title, actor, or director
            matching_titles = self.search_movies_advanced(search_query)
            if matching_titles:
                merged = merged[merged['title'].isin(matching_titles)]
            else:
                # Fallback to simple title search
                merged = merged[merged['title'].str.contains(search_query, case=False, na=False, regex=False)]
        
        # Filter by rating
        if min_rating > 0:
            merged = merged[merged['vote_average'] >= min_rating]
        
        # Remove movies without posters (only if poster_path column exists)
        if 'poster_path' in merged.columns:
            merged = merged[merged['poster_path'].notna()]
            merged = merged[merged['poster_path'] != '']
        else:
            # If no poster_path, add empty column as fallback
            merged['poster_path'] = ''
        
        # Sort
        if sort_by == "popularity" and 'popularity' in merged.columns:
            merged = merged.sort_values('popularity', ascending=False)
        elif sort_by == "rating":
            merged = merged.sort_values('vote_average', ascending=False)
        elif sort_by == "recent" and 'release_date' in merged.columns:
            merged = merged.sort_values('release_date', ascending=False)
        
        return merged.head(limit)
    
    def search_movies(self, query, limit=50):
        """Search for movies by partial title match"""
        if not query or len(query) == 0:
            return []
        query_lower = query.lower()
        
        matches = []
        for title in self.title_to_idx.keys():
            if query_lower in title:
                matches.append(title)
                if len(matches) >= limit:
                    break
        
        return matches
