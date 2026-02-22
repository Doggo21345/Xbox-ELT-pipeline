import unittest
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

class TestSilhouetteScore(unittest.TestCase):
    def test_silhouette_score_above_threshold(self):
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.60, random_state=0)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        self.assertGreater(score, 0.7)

if __name__ == '__main__':
    unittest.main()