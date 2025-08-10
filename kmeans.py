import numpy as np

class KMeans:

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(42)
        #Used to select k indices randomly from len(X)
        #Pick any k=3 indices from X, eg: 0, 8, 13
        random_idices = np.random.choice(len(X), self.k, replace=False)
        #Pick their values for centroids. self.centroids = [X[0],X[8],X[13]]
        self.centroids = X[random_idices]
        '''
            self.centroids = [[3,2,1,4],
                              [1,3,2,4],
                              [4,1,2,2]]
        '''
        
        for _ in range(self.max_iters):
            
            '''
                X[:, None] => Add an extra dimension/axis to X such that
                [[2,1,3,4],
                [1,0,2,3],
                [1,2,3,4]]... becomes
                [[[2,1,3,4]],
                 [[1,0,2,3]],
                 [[1,2,3,4]]]...    

                 X[:,None]-self.centroids =
                 [ [[2,1,3,4]]-[[3,1,2,4],[1,3,2,4],[4,1,2,2]],     [[[-1,0,1,0],[1,-2,1,0],[-2,0,1,2]],
                   [[1,0,2,3]]-[[3,1,2,4],[1,3,2,4],[4,1,2,2]],  =>  [[-2,-1,0,-1],[0,-3,0,-1],[-3,-1,0,1]],
                   [[1,2,3,4]]-[[3,1,2,4],[1,3,2,4],[4,1,2,2]]       [[-2,1,1,0],[0,-1,1,0],[-3,1,1,2]]]
                ]

                np.linarg.norm calculates euclidean distance of each element
                [[sqrt((-1)^2+0+1^2+0), sqrt(--), sqrt(--)],     [[1.41, 2.3, 2],
                 [--]                                         =>  [2, 1, 2.3].
                 [--]]                                            [2.3, 1.41, 3.4]]

                 np.argmin(distances, axis=1) will calculate the minimum value across each column and return its index
                 self.labels = [0, 1, 1], here 0 means it is closest to the first centroid
            '''
            
            distances = np.linalg.norm(X[:,None]-self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
            
            #calculating new centroids
            new_centroids = []
            for i in range(self.k):
                #if there are any points whose self.labels = i, then calculate their mean and assign it as the new centroid
                points = X[self.labels==i]
                if len(points) > 0:
                    new_centroids.append(points.mean(axis=0))
                #if there are no points whose labels = i, then assign the previous self.centroids[i] as the new centroid
                else:
                    new_centroids.append(self.centroids[i])
            
            new_centroids = np.array(new_centroids) 
            #if there is no difference in previous and new centroids, break
            if np.allclose(self.centroids, new_centroids):
                break
            #else assign new_centroids to self.centroids
            self.centroids = new_centroids
            
        
    def predict(self, X):
        distances = np.linalg.norm(X[:,np.newaxis] - self.centroids, axis=2) #X[:,np.newaxis] same as X[:,None]
        return np.argmin(distances, axis=1)
        
    