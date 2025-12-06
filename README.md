# 🎬 CineMatch - AI-Powered Movie Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

An intelligent movie recommendation system powered by **TF-IDF** and **Content-Based Filtering** algorithms, featuring a beautiful interactive interface with movie posters and personalized recommendations.

![CineMatch Demo](https://via.placeholder.com/800x400/1e1e1e/ffffff?text=CineMatch+Demo)

## ✨ Features

- 🎯 **Personalized Recommendations** - Get movie suggestions based on your favorites
- 🖼️ **Beautiful Poster Grid** - Interactive movie poster interface
- ❤️ **Heart-Based Favorites** - Click hearts to add movies to your collection
- 🎭 **Mood-Based Filtering** - Find movies by mood (Cozy, Mind-bending, Feel-good, etc.)
- 🔍 **Smart Search** - Search by title, actor, director, or keywords with fuzzy matching
- ⭐ **Rating Filters** - Filter by minimum rating and popularity
- 📊 **Advanced Algorithms** - TF-IDF vectorization with cosine similarity

## 🚀 Live Demo

**[Try CineMatch Live](https://your-app-url.streamlit.app)** *(Update this after deployment)*

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML/AI**: scikit-learn (TF-IDF, Cosine Similarity)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Dataset**: TMDB + IMDB Movies (127K+ movies)

## 📊 Algorithm Highlights

### Content-Based Filtering (TF-IDF)
- Uses **TF-IDF vectorization** to understand movie features
- Analyzes plot descriptions, genres, directors, cast, and keywords
- **Weighted features** for optimal recommendations:
  - Genres: 7x weight
  - Directors: 7x weight
  - Cast: 2x weight
  - Keywords: 3x weight
  - Overview: 1x weight

### Smart Search
- Fuzzy matching with word-by-word analysis
- Searches across titles, actors, directors, and plot descriptions
- Intelligent scoring system for relevance ranking

## 📁 Project Structure

```
Movie_recc/
├── app.py                          # Streamlit web app
├── recommendation_engine.py        # Core recommendation algorithm
├── movies_cleaned_full.csv         # Cleaned movie dataset
├── movies_preprocessed_model.csv   # Preprocessed features for ML
├── requirements.txt                # Python dependencies
├── Untitled.ipynb                  # Data analysis & model building
├── recommendation_cache/           # Cached TF-IDF matrices
│   ├── features.npz
│   └── cosine_similarity.pkl
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

## 🎯 How It Works

1. **Browse Movies** - Explore 127K+ quality movies (rating ≥ 6.0)
2. **Add Favorites** - Click ❤️ on movie posters to build your collection
3. **Get Recommendations** - Algorithm analyzes your taste and suggests similar movies
4. **Refine Results** - Filter by mood, rating, and search for specific titles

## 📊 Dataset Stats

- **Total Movies**: 127,475 quality films
- **Rating Range**: 6.0 - 10.0 (only high-quality movies)
- **Features**: 50,000+ TF-IDF features
- **Genres**: 20+ categories
- **Time Range**: Classic to modern cinema

## 🎨 Mood Categories

- 🛋️ Cozy
- 💔 Heart-breaking
- 🧠 Mind-bending
- 😊 Feel-good
- 🌑 Dark
- 🎨 Slow & Beautiful
- ⚔️ Epic Adventure
- 📼 Nostalgic
- ⚡ Intense & Gripping
- 🎪 Quirky & Weird
- 💪 Inspiring
- 👻 Scary

## 🔧 Local Installation

### 1. Clone and Install Dependencies
```bash
# Clone the repository
git clone https://github.com/fadhelcherif/MovieReccomendation.git
cd MovieReccomendation

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data Files (IMPORTANT!)
⚠️ **The CSV data files are too large for GitHub (450MB+)**

**You need to download them separately:**
- `movies_cleaned_full.csv` (96 MB)
- `movies_preprocessed_model.csv` (70 MB)  
- `TMDB IMDB Movies Dataset.csv` (264 MB)

**Download Link**: *[Upload your files to Google Drive/Dropbox and add link here]*

Place the CSV files in the project root directory.

### 3. Run the App
```bash
streamlit run app.py
```

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn
- scipy
- Pillow

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Fadhel Cherif**
- GitHub: [@fadhelcherif](https://github.com/fadhelcherif)

## 🙏 Acknowledgments

- Dataset: TMDB & IMDB
- Movie Posters: The Movie Database (TMDB)
- Framework: Streamlit Community

## 📧 Contact

Have questions or suggestions? Feel free to open an issue or reach out!

---

**Made with ❤️ using Streamlit and scikit-learn**
