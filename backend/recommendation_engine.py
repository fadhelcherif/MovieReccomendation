
import ast
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, data_path="movies_preprocessed_model_sample.csv"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.base_dir, data_path)
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        print(f"4c2 Loading dataset: {self.data_path}")
        self.model_df = pd.read_csv(self.data_path, low_memory=False)
        print(f"[DEBUG] Loaded DataFrame shape: {self.model_df.shape}")
        print(f"[DEBUG] Loaded DataFrame columns: {list(self.model_df.columns)}")
        print(f"[DEBUG] First 2 rows:\n{self.model_df.head(2).to_dict()}")
        self.model_df = self.model_df[self.model_df["title"].notna()].reset_index(drop=True)
        if "vote_average" in self.model_df.columns:
            self.model_df = self.model_df[self.model_df["vote_average"] >= 6.0]
        for col in ["genres", "keywords", "cast", "directors"]:
            if col in self.model_df.columns:
                self.model_df[col] = self.model_df[col].apply(self._parse_list_col)
        self.model_df["overview"] = self.model_df["overview"].fillna("")
        print(f"[DEBUG] DataFrame after cleaning shape: {self.model_df.shape}")
        print(f"[DEBUG] DataFrame after cleaning columns: {list(self.model_df.columns)}")
        print(f"[DEBUG] DataFrame after cleaning first 2 rows:\n{self.model_df.head(2).to_dict()}")
        self._build_features()

    # ---------------- helpers ----------------
    def _parse_list_col(self, x):
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(str(x))
        except (ValueError, SyntaxError):
            return [i.strip() for i in str(x).split(",")]

    def _clean(self, text):
        return str(text).replace(" ", "").lower()

    def _create_soup(self, row):
        genres = " ".join(map(self._clean, row.get("genres", [])))
        directors = " ".join(map(self._clean, row.get("directors", [])))
        cast = " ".join(map(self._clean, row.get("cast", [])[:5]))
        keywords = " ".join(map(self._clean, row.get("keywords", [])[:10]))
        overview = self._clean(row.get("overview", ""))

        return f"{genres} {directors} {cast} {keywords} {overview}"

    # ---------------- features ----------------
    def _build_features(self):
        self.model_df["soup"] = self.model_df.apply(self._create_soup, axis=1)

        self.tfidf = TfidfVectorizer(stop_words="english", max_features=30000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.model_df["soup"])

        self.title_to_idx = pd.Series(
            self.model_df.index,
            index=self.model_df["title"].str.lower()
        ).to_dict()

    # ---------------- MAIN RECOMMENDATION ----------------
    def get_recommendations_from_favorites(self, favorites, top_n=10):

        all_recs = []

        for title in favorites:
            title = title.lower()

            if title not in self.title_to_idx:
                continue

            idx = self.title_to_idx[title]

            sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
            similar_idx = sim.argsort()[::-1][1:top_n+1]

            all_recs.append(self.model_df.iloc[similar_idx])

        if not all_recs:
            return pd.DataFrame()

        result = pd.concat(all_recs)
        result = result.drop_duplicates(subset=["title"])

        return result.head(top_n)

    # ---------------- FIX (THIS WAS MISSING) ----------------
    def get_movie_details(self, title):
        title = title.lower()

        if title not in self.title_to_idx:
            return None

        row = self.model_df.iloc[self.title_to_idx[title]]

        return {
            "title": row["title"],
            "genres": row.get("genres", ""),
            "directors": row.get("directors", "")
        }
