import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KMeans:

    def __init__(self,K=5,max_iter=1000,plot_steps=False) :
        self.K = K
        self.max_iter = max_iter
        self.plot_steps=plot_steps

        # List of sample Indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        #Store the centers(Mean vector) for each cluster
        self.centroid =[]

    def predict(self,X):
        self.X=X
        self.n_samples, self.n_features = X.shape

        #initialize Centroids
        random_sample_indices = np.random.choice(self.n_samples,self.K,replace=False)
        self.centroid = [self.X[idx] for idx in random_sample_indices]

        #optimize clusters
        for _ in range(self.max_iter):
            #assign the samples to closest centroids
            self.clusters = self._create_clusters(self.centroid)

            if self.plot_steps:
                self.plot()

            #calculate new centoroids from the cluster
            centroids_old = self.centroid
            self.centroid = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old,self.centroid):
                break
            if self.plot_steps:
                self.plot()

        #Classify the samples of index of their clusters
        return self._get_cluster_labels(self.clusters)


    def _create_clusters(self,centroids):
        #assign samples to closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self,sample,centroids):
        #distance of the current sample to closest centroid
        distances = [euclidean_distance(sample,point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
        
    def _get_centroids(self,clusters):
        #assign the mean value of the clusters to the centroids
        centroids = np.zeros((self.K,self.n_features))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster],axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self,centroids_old,centroids):
        #distances and old and new centroids for all centroids
        distances = [euclidean_distance(centroids_old[i],centroids[i]) for i in range(self.K)]
        return sum(distances)==0


    def _get_cluster_labels(self,clusters):
        #Each sample will get the label of the cluster it is assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx]=cluster_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(8,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroid:
            ax.scatter(*point,marker="x",color="black",linewidth=2)
        
        plt.show()
        

if __name__ =="__main__":
    np.random.seed(422)
    from sklearn.datasets import make_blobs

    X,y = make_blobs(centers=3,n_samples=500,n_features=2,shuffle=True,random_state=455)

    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k= KMeans(K=clusters,max_iter=150,plot_steps=True)
    y_pred= k.predict(X)

    k.plot()