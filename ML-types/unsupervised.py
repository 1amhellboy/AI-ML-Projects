import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# 1. Load dataset
iris = load_iris()

# 2. Take only 2 features (sepal length, sepal width) for easy visualization
X = iris.data[:, :2]

# 3. Apply KMeans clustering (unsupervised → no labels used)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 4. Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap="viridis", s=50)
plt.title("Unsupervised Clustering (Iris dataset)")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()
