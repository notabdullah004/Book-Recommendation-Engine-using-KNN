# 1. 📦 Imports & Data Load
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Data: Book-Crossings ratings and book metadata
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', encoding='latin-1',
                      names=['user', 'isbn', 'rating'], skiprows=1, low_memory=False)
books   = pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1',
                      names=['isbn','title','author','year','publisher'], skiprows=1, low_memory=False)

# 2. 🧹 Clean: filter out low-activity users/books
user_counts = ratings['user'].value_counts()
book_counts = ratings['isbn'].value_counts()
filtered = ratings[
    ratings['user'].isin(user_counts[user_counts >= 200].index) &
    ratings['isbn'].isin(book_counts[book_counts >= 100].index)
]

# 3. 🔗 Merge with titles and drop duplicate titles
df = filtered.merge(books[['isbn','title']], on='isbn')
df = df.drop_duplicates(['user','title'])

# 4. 🔢 Create user–book matrix
matrix = df.pivot(index='title', columns='user', values='rating').fillna(0)
sparse_matrix = csr_matrix(matrix.values)

# 5. 🔍 Train KNN (cosine distance)
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(sparse_matrix)

# 6. 👉 get_recommends function
def get_recommends(book_title, n_recs=5):
    if book_title not in matrix.index:
        raise ValueError(f"'{book_title}' not in dataset")
    idx = matrix.index.get_loc(book_title)
    distances, indices = model.kneighbors(
        sparse_matrix[idx], n_neighbors=n_recs+1
    )
    recs = []
    for i, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):
        recs.append([matrix.index[i], float(dist)])
    return [book_title, recs]

# 7. ✅ Test it
if __name__ == "__main__":
    test = "The Queen of the Damned (Vampire Chronicles (Paperback))"
    result = get_recommends(test, n_recs=5)
    print(result)
