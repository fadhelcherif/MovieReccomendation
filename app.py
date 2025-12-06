"""
🎬 Movie Recommendation System - Interactive Poster-Based Interface
Beautiful grid interface with movie posters and heart-based favorites
"""

import streamlit as st
import pandas as pd
from recommendation_engine import MovieRecommender
from PIL import Image

# Page configuration
st.set_page_config(
    page_title=" CineMatch",
    page_icon="logo.png ",
  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for movie poster grid and heart animations
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    
    /* Logo in top-left corner */
    [data-testid="stSidebarNav"] {
        background-image: url('app/static/logo.png');
        background-repeat: no-repeat;
        background-position: 20px 20px;
        background-size: 100px;
        padding-top: 50px;
        padding-bottom: 20px;
    }
    
    /* Movie Card Styling */
    .movie-card {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        background: #1e1e1e;
        height: 100%;
        cursor: pointer;
    }
    
    .movie-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(255, 75, 75, 0.3);
    }
    
    .movie-poster {
        width: 100%;
        height: 400px;
        object-fit: cover;
        display: block;
    }
    
    .movie-info {
        padding: 12px;
        background: linear-gradient(to top, #000000, transparent);
        position: absolute;
        bottom: 0;
        width: 100%;
        color: white;
    }
    
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    
    .movie-rating {
        font-size: 14px;
        color: #ffd700;
        margin-top: 4px;
    }
    
    /* Heart Button */
    .heart-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        font-size: 28px;
        cursor: pointer;
        z-index: 10;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        border: none;
    }
    
    .heart-btn:hover {
        transform: scale(1.2);
        background: rgba(0, 0, 0, 0.8);
    }
    
    /* Filter Section */
    .filter-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        color: white;
    }
    
    /* Favorites Bar */
    .favorites-bar {
        background: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .favorite-chip {
        display: inline-block;
        background: #FF4B4B;
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        font-weight: bold;
        border-radius: 25px;
        padding: 12px 24px;
        border: none;
        font-size: 16px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8B8B 100%);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
        transform: translateY(-2px);
    }
    
    /* Header */
    h1 {
        background: linear-gradient(135deg, #FF4B4B 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    
    </style>
""", unsafe_allow_html=True)

# Cache the recommender initialization to prevent reloading
@st.cache_resource
def load_recommender():
    """Load and cache the movie recommender"""
    return MovieRecommender()

# Cache movie data fetching for better performance
@st.cache_data(ttl=3600)
def get_filtered_movies(mood, min_rating, sort_by, search_query, num_display):
    """Cache filtered movie results for 1 hour"""
    _recommender = load_recommender()
    
    if mood != "All Moods":
        mood_name = mood.split(" ", 1)[1]
        return _recommender.filter_by_mood(
            mood=mood_name,
            limit=num_display,
            min_rating=min_rating
        )
    else:
        return _recommender.get_movies_with_posters(
            limit=num_display,
            min_rating=min_rating,
            sort_by=sort_by,
            search_query=search_query
        )

@st.cache_data(ttl=3600)
def get_recommendations_cached(favorites, num_recommendations):
    """Cache recommendation results"""
    _recommender = load_recommender()
    return _recommender.get_recommendations_from_favorites(
        list(favorites),
        top_n=num_recommendations
    )

@st.cache_data(ttl=3600)
def get_movie_details_cached(title):
    """Cache movie details lookup"""
    _recommender = load_recommender()
    return _recommender.get_movie_details(title)

# Initialize session state
if 'favorites' not in st.session_state:
    st.session_state.favorites = set()

if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False

# Load the cached recommender
with st.spinner('🎬 Loading movie database... Please wait...'):
    recommender = load_recommender()

# Helper function to build poster URL
def get_poster_url(poster_path, size="w500"):
    """Convert poster_path to full TMDB URL"""
    if pd.isna(poster_path) or poster_path == "":
        return "https://via.placeholder.com/500x750/1e1e1e/ffffff?text=No+Poster"
    if poster_path.startswith("http"):
        return poster_path
    return f"https://image.tmdb.org/t/p/{size}{poster_path}"

# Logo in sidebar
with st.sidebar:
    try:
        logo = Image.open("logo.png")
        st.image(logo, width=200)
    except:
        pass

# Header
st.markdown("<h1 style='text-align: center; font-size: 48px; margin-top: 0;'>Discover Your Next Favorite Movie</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 18px;'>Click the ❤️ on posters to add favorites, then get personalized recommendations!</p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - Filters and Controls
with st.sidebar:
    st.markdown("### 🎛️ Filters & Settings")
    
    # Mood filter
    mood_options = [
        "All Moods", 
        "🛋️ Cozy", 
        "💔 Heart-breaking", 
        "🧠 Mind-bending", 
        "😊 Feel-good", 
        "🌑 Dark", 
        "🎨 Slow & Beautiful",
        "⚔️ Epic Adventure",
        "📼 Nostalgic",
        "⚡ Intense & Gripping",
        "🎪 Quirky & Weird",
        "💪 Inspiring",
        "👻 Scary"
    ]
    selected_mood = st.selectbox(
        "🎭 Mood",
        options=mood_options,
        help="Filter movies by mood and atmosphere"
    )
    
    # Rating filter
    min_rating = st.slider(
        "⭐ Minimum Rating",
        min_value=0.0,
        max_value=10.0,
        value=6.0,
        step=0.5,
        help="Filter movies by minimum rating"
    )
    
    # Number of movies to display
    num_display = st.slider(
        "📊 Movies to Display",
        min_value=20,
        max_value=200,
        value=40,  # Reduced from 60 for faster loading
        step=20,
        help="Number of movies to show in the grid"
    )
    
    # Sort options
    sort_by = st.selectbox(
        "🔽 Sort By",
        options=["popularity", "rating", "recent"],
        format_func=lambda x: {"popularity": "🔥 Popularity", "rating": "⭐ Rating", "recent": "📅 Recent"}.get(x, x)
    )
    
    st.markdown("---")
    
    # Recommendations settings
    st.markdown("### 🎯 Recommendation Settings")
    
    num_recommendations = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=30,
        value=12,
        step=1
    )
    
    use_year_filter = st.checkbox("📅 Filter by similar release years", value=False)
    
    st.markdown("---")
    
    # Favorites count
    st.markdown(f"### ❤️ Favorites ({len(st.session_state.favorites)})")
    
    if st.session_state.favorites:
        for idx, fav in enumerate(list(st.session_state.favorites)):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(fav)
            with col2:
                if st.button("❌", key=f"remove_sidebar_{idx}_{hash(fav)}"):
                    st.session_state.favorites.discard(fav)
                    st.rerun()
        
        if st.button("🗑️ Clear All Favorites", key="clear_all_btn"):
            st.session_state.favorites.clear()
            st.session_state.show_recommendations = False
            st.rerun()
        
        st.markdown("---")
        
        if st.button("🎬 GET RECOMMENDATIONS", key="get_rec_sidebar"):
            st.session_state.show_recommendations = True
            st.rerun()
    else:
        st.info("Add movies to favorites by clicking the ❤️ on posters!")

# Main content area
if st.session_state.show_recommendations and st.session_state.favorites:
    # Show recommendations
    st.markdown("## 🎯 Your Personalized Recommendations")
    
    # Get recommendations first to analyze
    try:
        # Get recommendations (cached for speed!)
        results = get_recommendations_cached(
            tuple(st.session_state.favorites),  # Convert to tuple for caching
            num_recommendations
        )
        
        # Analyze favorite movies to create summary
        favorite_list = list(st.session_state.favorites)
        favorite_genres = set()
        favorite_directors = set()
        
        for fav in favorite_list:
            movie_details = get_movie_details_cached(fav)
            if movie_details:
                genres = movie_details['genres'].split(', ')
                favorite_genres.update(genres[:3])  # Top 3 genres
                directors = movie_details['directors'].split(', ')
                favorite_directors.update(directors[:2])  # Top 2 directors
        
        # Create explanation
        st.markdown("---")
        st.markdown("### 📋 Your Selections")
        # Display chosen movies
        if len(favorite_list) <= 5:
            movies_text = ", ".join([f"**{m}**" for m in favorite_list])
        else:
            movies_text = ", ".join([f"**{m}**" for m in favorite_list[:5]]) + f" and **{len(favorite_list)-5} more**"
        
        st.markdown(f"You chose: {movies_text}")
        # Overall explanation
        explanation_parts = []
        if favorite_genres:
            top_genres = list(favorite_genres)[:3]
            explanation_parts.append(f"you enjoy **{', '.join(top_genres)}** genres")
        if favorite_directors:
            top_directors = list(favorite_directors)[:2]
            explanation_parts.append(f"you like films by **{', '.join(top_directors)}**")
        
        if explanation_parts:
            explanation = " and ".join(explanation_parts)
            st.info(f"💡 **Why these recommendations?** Based on your selections, we found that {explanation}. We've matched you with similar movies that share these elements!")
        else:
            st.info(f"💡 **Why these recommendations?** We analyzed the themes, styles, and characteristics of your favorite movies to find similar films you'll love!")
        
        st.markdown("---")

        if st.button("🔙 Back to Browse", key="back_browse_main"):
            st.session_state.show_recommendations = False
            st.rerun()
        
        # Get full data with posters for recommendations
        rec_movies = recommender.full_df[recommender.full_df['title'].isin(results['title'].tolist())]
        
        st.markdown(f"### 🎬 Top {len(results)} Recommendations for You")
        
        # Display in grid
        cols_per_row = 4
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(results):
                    row = results.iloc[i + j]
                    movie_data = rec_movies[rec_movies['title'] == row['title']]
                    
                    with col:
                        # Get poster
                        poster_path = movie_data['poster_path'].values[0] if len(movie_data) > 0 and 'poster_path' in movie_data.columns else ""
                        poster_url = get_poster_url(poster_path)
                        
                        # Display movie card
                        st.markdown(f"""
                            <div class="movie-card">
                                <img src="{poster_url}" class="movie-poster" alt="{row['title']}">
                                <div class="movie-info">
                                    <p class="movie-title">{row['title']}</p>
                                    <p class="movie-rating">⭐ {row['vote_average']:.1f} | Match: {row['match_%']:.0f}%</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"🎭 {row['genres'][:50]}...")
        
        # Download option
        st.markdown("---")
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations as CSV",
            data=csv,
            file_name="my_movie_recommendations.csv",
            mime="text/csv"
        )
            
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")

else:
    # Browse movies mode
    st.markdown("## 🎬 Browse Movies")
    
    # Search bar in main area
    search_query = st.text_input(
        "🔍 Search for movies...",
        placeholder="Search by title, actor, director, or keywords (e.g., 'dark knight', 'nolan', 'dicaprio')...",
        help="Smart search: finds movies even with partial words. Try 'batman', 'nolan', or 'dark knight'",
        key="main_search"
    )
    
    # Only search if query is longer than 1 character or empty
    if search_query and len(search_query) < 2:
        st.info("💡 Type at least 2 characters to search")
        search_query = ""  # Reset to show all movies
    
    st.markdown("---")
    
    # Get movies with posters (cached for speed!)
    movies = get_filtered_movies(
        mood=selected_mood,
        min_rating=min_rating,
        sort_by=sort_by,
        search_query=search_query,
        num_display=num_display
    )
    
    # Show appropriate message
    if selected_mood != "All Moods":
        mood_name = selected_mood.split(" ", 1)[1]
        st.markdown(f"**{len(movies)} movies for mood: {selected_mood}** | Rating ≥ {min_rating}")
    elif search_query and len(search_query) > 0:
        st.markdown(f"**Found {len(movies)} movies matching '{search_query}'** | Rating ≥ {min_rating}")
    else:
        st.markdown(f"**Showing {len(movies)} movies** | Rating ≥ {min_rating} | Click ❤️ to add to favorites")
    
    if len(movies) == 0:
        st.warning("No movies found with the current filters. Try adjusting the rating threshold.")
    else:
        # Display movies in grid
        cols_per_row = 5
        for i in range(0, len(movies), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(movies):
                    movie = movies.iloc[i + j]
                    
                    with col:
                        # Get poster URL
                        poster_url = get_poster_url(movie['poster_path'])
                        
                        # Create unique key for this movie
                        movie_key = f"{movie['title']}_{i}_{j}"
                        is_favorite = movie['title'] in st.session_state.favorites
                        
                        # Display movie poster
                        st.markdown(f"""
                            <div class="movie-card">
                                <img src="{poster_url}" class="movie-poster" alt="{movie['title']}" loading="lazy">
                                <div class="movie-info">
                                    <p class="movie-title">{movie['title'][:40]}{'...' if len(movie['title']) > 40 else ''}</p>
                                    <p class="movie-rating">⭐ {movie['vote_average']:.1f}</p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Heart button - INSTANT with rerun for immediate update
                        heart_label = "💔 Remove" if is_favorite else "❤️ Favorite"
                        if st.button(heart_label, key=f"heart_{i}_{j}_{hash(movie['title'])}", use_container_width=True):
                            if is_favorite:
                                st.session_state.favorites.discard(movie['title'])
                            else:
                                st.session_state.favorites.add(movie['title'])
                            st.rerun()  # Immediate update so you see it added
                        
                        # Show genre
                        genres = movie['genres']
                        if isinstance(genres, list):
                            genre_str = ', '.join(genres[:2])
                        else:
                            genre_str = str(genres)[:30]
                        st.caption(f"🎭 {genre_str}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Movies", f"{len(movies):,}" if 'movies' in locals() else "0")
with col2:
    st.metric("Favorites Selected", len(st.session_state.favorites))
with col3:
    avg_rating = movies['vote_average'].mean() if 'movies' in locals() and len(movies) > 0 else 0
    st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
