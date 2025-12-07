/*
===========================================================
                   CineMatch – Movie Recommendation System
===========================================================

[Streamlit App Badge]

CineMatch is a movie recommendation system based on TF-IDF and 
content-based filtering. It provides personalized suggestions 
through a simple interface built around movie posters, search, 
and filtering options.

-----------------------------------------------------------
Features
-----------------------------------------------------------
- Personalized recommendations
- Poster-based browsing
- Favorites list
- Mood filters (Cozy, Mind-bending, Feel-good, etc.)
- Search by title, actor, director, or keywords
- Rating filters
- TF-IDF + cosine similarity engine

-----------------------------------------------------------
Live Demo
-----------------------------------------------------------
The live app will be available after deployment:
https://fadhelcherif-moviereccomendation-app-fjqapm.streamlit.app/
-----------------------------------------------------------
Technology Stack
-----------------------------------------------------------
- Streamlit
- Python, scikit-learn
- Pandas, NumPy
- Matplotlib
- TMDB / IMDB datasets

-----------------------------------------------------------
Project Structure
-----------------------------------------------------------
Movie_recc/
    app.py
    recommendation_engine.py
    movies_cleaned_full.csv
    movies_preprocessed_model.csv
    requirements.txt
    .streamlit/

-----------------------------------------------------------
How It Works
-----------------------------------------------------------
1. Browse through the movie catalogue
2. Add movies to favorites
3. Get recommendations based on similarity
4. Refine results with filters

-----------------------------------------------------------
Installation
-----------------------------------------------------------
1. Clone and install dependencies:

    git clone https://github.com/fadhelcherif/MovieReccomendation.git
    cd MovieReccomendation
    pip install -r requirements.txt

2. Add dataset files (download separately):

    - movies_cleaned_full.csv
    - movies_preprocessed_model.csv
    - TMDB/IMDB dataset (optional)

   Place them in the project root directory.

3. Run the application:

    streamlit run app.py

   The app will run at:
   http://localhost:8501

-----------------------------------------------------------
Requirements
-----------------------------------------------------------
Python 3.8+ with libraries listed in requirements.txt

-----------------------------------------------------------
Author
-----------------------------------------------------------
Fadhel Cherif
GitHub: https://github.com/fadhelcherif

-----------------------------------------------------------
License
-----------------------------------------------------------
MIT License

===========================================================
*/
