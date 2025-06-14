import tmdbsimple as tmdb
from sentence_transformers import SentenceTransformer
import faiss
import pickle

tmdb.API_KEY = "65840f206e1e8eef41ff1fab532ad49b"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PAGES      = 50
INDEX_PATH = "movies.index"
META_PATH  = "movies_meta.pkl"
def fetch_popular(pages: int):
    movies = []
    for p in range(1, pages + 1):
        res = tmdb.Discover().movie(page=p, sort_by="popularity.desc")
        movies += res["results"]
    return movies

def main():
    print("загрузка")
    movies = fetch_popular(PAGES)
    texts = []
    for m in movies:
        parts = [
            m.get("title", ""),
            m.get("original_title", ""),
            m.get("overview", "")
        ]
        texts.append(" ".join(filter(None, parts)))
    print("эмбеддинги")
    model = SentenceTransformer(MODEL_NAME)
    embs  = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print("FAISS-индекс")
    faiss.normalize_L2(embs)
    d     = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    print(f"индекс в «{INDEX_PATH}» и метаданные в «{META_PATH}»…")
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(movies, f)
    print("done")

if __name__ == "__main__":
    main()