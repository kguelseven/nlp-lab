from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sentences = [
    "I love cats",
    "I love dogs",
    "I love cars",
    "I like animals",
    "I enjoy movies",
    "I like beer",
    "This is terrible",
    "I hate this",
    "I loathe this juice"
]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)

kmeans = KMeans(n_clusters=2).fit(embeddings)
print(kmeans.labels_)

# reducer = umap.UMAP()
reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, random_state=42)
points = reducer.fit_transform(embeddings)

labels = [0, 0, 0, 0, 0, 0, 1, 1, 1]
plt.scatter(points[:,0], points[:,1], c=labels, cmap='coolwarm')

for i, txt in enumerate(sentences):
    plt.text(points[i,0], points[i,1], txt)

plt.show()