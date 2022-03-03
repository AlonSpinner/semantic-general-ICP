import matplotlib.pyplot as plt
import numpy as np
import utils.plotting2D
from sklearn.neighbors import NearestNeighbors

class cluster2:

    #class attributes
    markersList = ["o","v","s","D","4","*",">","<","2","P"] #10 classes max
    colors = np.array([[0.244972, 0.287675, 0.53726 , 1.      ],
       [0.709898, 0.868751, 0.169257, 1.      ],
       [0.147607, 0.511733, 0.557049, 1.      ],
       [0.993248, 0.906157, 0.143936, 1.      ],
       [0.281412, 0.155834, 0.469201, 1.      ],
       [0.20803 , 0.718701, 0.472873, 1.      ],
       [0.430983, 0.808473, 0.346476, 1.      ],
       [0.190631, 0.407061, 0.556089, 1.      ],
       [0.267004, 0.004874, 0.329415, 1.      ],
       [0.119699, 0.61849 , 0.536347, 1.      ]])
    
    def __init__(self,points = None, covariances = None, pointLabels = None):
        '''
        each landmark is dictionary:
        {x - float,
        y - float,
        classLabel - string,
        covaraince - 2x2 float matrix,
        index - integer}. 
        '''

        #instance attributes
        self.points = np.array([]) #numpy array of points (m,2,1)
        self.covariances = np.array([]) #numpy array of covariances (m,2,2)
        self.pointLabels = [] # list of strings
     
        self.classes = [] #list of all class labels
        
        if not np.empty(points):
            self.addPoints(points, covariances, pointLabels)

    def addPoints(self,points, covariances = None, pointLabels = None):
        assert len(covariances) == len(points) == len(pointLabels)
        
        if self.points.size == 0:
            self.points = points
        else:
            self.points = np.vstack((self.points,points))

        if covariances is not None:
            if self.covariances.size == 0:
                self.covariances = covariances
            else:
                self.covariances = np.vstack((self.covariances,covariances))
            
        if pointLabels is not None:
            self.pointLabels.extend(pointLabels)
        
        self.defineClassesFromPointLabels(add = pointLabels)

    def fillClusterRandomly(self, N, xrange, yrange ,sigmarange = None,classes = None):
        # N - amount of Points
        # classLabels - list of strings
        # xrange, yrange - tuples
        
        x = np.random.uniform(xrange[0],xrange[1],(N,1))
        y = np.random.uniform(yrange[0],yrange[1],(N,1))
        points = np.hstack((x,y)).reshape(-1,2,1)

        if sigmarange is not None:
            covariances = np.zeros((N,2,2))
            for i in range(N):
                rootcov = np.random.uniform(low=sigmarange[0], high=sigmarange[1], size=(2,2))
                covariances[i] = rootcov @ rootcov.T #enforce symmetric and positive definite: https://mathworld.wolfram.com/PositiveDefiniteMatrix.html
        else:
            cov = None

        if classes is not None:
            pointLabels = np.random.choice(classes, N)  
        else:
            pointLabels = None

        self.addPoints(points, covariances, pointLabels)

    #goes over all Points to find classLabels.
    def defineClassesFromPointLabels(self, add: list = None):
        if add is not None: #list of pointLabels that are added
            newclasses = list(set(add)) #find unique new classes
            self.classes = list(set(self.classes + newclasses))
        else:
            self.classes = list(set(self.pointLabels))
            if len(self.classes) > 10:
                raise Exception("no more than 10 classes are premited. Not enough distinguishable markers in matplotlib")

    def exportSemantics(self):
        semantics = {
                    "classLabel": self.classes,
                    "color": self.colors[:len(self.classes)],
                    "marker": self.markersList[:len(self.classes)]
                    }
        return semantics

    def plot(self,ax = None, plotIndex = False, plotCov = False,
                 markerSize = 10, markerColor = None, markerShape = None):
        if ax == None:
            ax, _ = utils.plotting2D.prepAxes()

        semanticColor = True if markerColor is None else False
        semanticShape = True if markerShape is None else False
            

        for ii, point in enumerate(self.points):
            kk = self.classes.index(self.pointLabels[ii]) #index of classLabel. used for shape and color

            if semanticColor:
                markerColor = self.colors[kk].reshape(1,-1) #color from semantics

            if semanticShape:
                markerShape = self.markersList[kk] #shape from semantics

            cov = None if plotCov is False else self.covariances[ii]
            index = None if plotIndex is False else ii

            utils.plotting2D.point(ax, loc = point, cov = cov, 
                                index = index, 
                                markerColor = markerColor, 
                                markerShape = markerShape, 
                                markerSize = markerSize,
                                textColor = 'k')

    def transform(self, R: np.ndarray((2,2)), t: np.ndarray((2,1)), transformCov = True):
        #transforms Points locations and convarance matrices
        self.points = (R @ self.points.T + t).T
        if transformCov:
            self.covariances = (R @ self.covariances[:,None,:] @ R.T).squeeze()

    def findKnnInTarget(self,targetCluster, n = 1):        
        neigh = NearestNeighbors(n_neighbors = n )
        neigh.fit(targetCluster.points)
        distances, indicies = neigh.kneighbors(self.points, return_distance=True)
        return indicies, distances


    

    
