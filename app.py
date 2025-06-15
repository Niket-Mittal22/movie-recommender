from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

movie_embeddings = np.load(os.path.join(base_dir, "processed_data", "embedding_matrix.npy"))

import faiss

# Convert to float32
embedding_matrix_f32 = movie_embeddings.astype('float32')

# Create FAISS index
index_faiss = faiss.IndexFlatIP(embedding_matrix_f32.shape[1])  # IP = inner product = cosine if normalized
faiss.normalize_L2(embedding_matrix_f32)  # Normalize for cosine similarity
index_faiss.add(embedding_matrix_f32)


with open(os.path.join(base_dir, "processed_data", "final_dataset.pkl"), "rb") as f:
    movies = pickle.load(f)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')  # small and fast

from rapidfuzz import process

def get_best_title_match(user_input):
    choices = movies['title'].tolist()
    match = process.extractOne(user_input, choices, score_cutoff=80)
    return match[0] if match else None

def get_index_from_title(title):
    title = title.lower().strip()
    matches = movies[movies['title'].str.lower().str.strip() == title]
    if not matches.empty:
        return matches.index[0]
    else:
        return None
    
def get_movie_info(index):
    return movies.iloc[index]  # Return it

def get_movie_plot(index):
    return get_movie_info(index)['plot']

def get_movie_embedding(index):
    return get_movie_info(index)['embedding']


def get_query_embedding(text):
    query_embedding = model.encode(text)
    return query_embedding

def get_top_5_similar_movies(query_embedding, top_k=5):
    query_vector = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_vector)  # Needed for cosine similarity behavior
    _, indices = index_faiss.search(query_vector, top_k)
    top_movies = [movies.iloc[i] for i in indices[0]]
    return top_movies


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/plot-search', methods=['GET', 'POST'])
def plot_search():
    if request.method == 'POST':
        query = request.form['plot']
        embed = get_query_embedding(query)
        top5 = get_top_5_similar_movies(embed)
        return render_template('plot_search.html', top5 = top5)
    return render_template('plot_search.html')

@app.route('/similar-movie', methods=['GET', 'POST'])
def similar_movie():
    if request.method == 'POST':
        title = request.form['title']
        matched_title = get_best_title_match(title)
        index = get_index_from_title(matched_title)
        embed = get_movie_embedding(index)
        top5 = get_top_5_similar_movies(embed)
        return render_template('similar_movie.html', top5 = top5)
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use PORT from env or default to 8000
    app.run(debug=False, host="0.0.0.0", port=port)