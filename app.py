from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

# === SAFELY LOAD FILES USING ABSOLUTE PATHS ===
base_dir = os.path.dirname(os.path.abspath(__file__))
embedding_path = os.path.join(base_dir, "processed_data", "embedding_matrix.npy")
dataset_path = os.path.join(base_dir, "processed_data", "final_dataset.pkl")

movie_embeddings = np.load(embedding_path)

import faiss

# Convert to float32 and create FAISS index
embedding_matrix_f32 = movie_embeddings.astype('float32')
faiss.normalize_L2(embedding_matrix_f32)
index_faiss = faiss.IndexFlatIP(embedding_matrix_f32.shape[1])
index_faiss.add(embedding_matrix_f32)

with open(dataset_path, "rb") as f:
    movies = pickle.load(f)

from sentence_transformers import SentenceTransformer
from rapidfuzz import process

model = SentenceTransformer('all-mpnet-base-v2')

def get_best_title_match(user_input):
    choices = movies['title'].tolist()
    match = process.extractOne(user_input, choices, score_cutoff=80)
    return match[0] if match else None

def get_index_from_title(title):
    title = title.lower().strip()
    matches = movies[movies['title'].str.lower().str.strip() == title]
    return matches.index[0] if not matches.empty else None

def get_movie_info(index):
    return movies.iloc[index]

def get_movie_embedding(index):
    return get_movie_info(index)['embedding']

def get_query_embedding(text):
    return model.encode(text)

def get_top_5_similar_movies(query_embedding, top_k=5):
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_vector)
    _, indices = index_faiss.search(query_vector, top_k)
    return [movies.iloc[i] for i in indices[0]]

# === FLASK APP ===
app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/plot-search', methods=['GET', 'POST'])
def plot_search():
    if request.method == 'POST':
        query = request.form['plot']
        embed = get_query_embedding(query)
        top5 = get_top_5_similar_movies(embed)
        return render_template('plot_search.html', top5=top5)
    return render_template('plot_search.html')

@app.route('/similar-movie', methods=['GET', 'POST'])
def similar_movie():
    if request.method == 'POST':
        title = request.form['title']
        matched_title = get_best_title_match(title)
        index = get_index_from_title(matched_title)
        embed = get_movie_embedding(index)
        top5 = get_top_5_similar_movies(embed)
        return render_template('similar_movie.html', top5=top5)
    return render_template('similar_movie.html')

@app.route('/movie-suggestions')
def movie_suggestions():
    query = request.args.get('q', '').strip()
    if len(query) < 2:
        return jsonify([])
    all_titles = movies['title'].tolist()
    matches = process.extract(query, all_titles, limit=5, score_cutoff=70)
    suggestions = [{'title': match[0]} for match in matches]
    return jsonify(suggestions)

# === RENDER FRIENDLY ENTRYPOINT ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
