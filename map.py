import matplotlib.pyplot as plt
import numpy as np
import utils.plotting

class map:

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
    
    def __init__(self,landmarks = []):
        '''
        each landmark is dictionary:
        {x - float,
        y - float,
        classLabel - string,
        covaraince - 2x2 float matrix,
        index - integer}. 
        '''

        #instance attributes
        self.landmarks = []
        self.classLabels = [] #will be filled laters
        self.addLandmarks(landmarks)

    def fillMapRandomly(self, N, classLabels, xrange, yrange, sigmarange = None):
        # N - amount of landmarks
        # classLabels - list of strings
        # xrange, yrange - tuples
        landmarks = [None] * N

        for ii in range(N):
            xy = np.array([np.random.uniform(xrange[0],xrange[1]),
                           np.random.uniform(yrange[0],yrange[1])])  

            if sigmarange is not None:
                rootcov = np.random.uniform(low=sigmarange[0], high=sigmarange[1], size=(2,2))
                cov = rootcov @ rootcov.T #enforce symmetric and positive definite: https://mathworld.wolfram.com/PositiveDefiniteMatrix.html
            else:
                cov = None
                      
            landmarks[ii] = landmark(xy, 
                                     classLabel = np.random.choice(classLabels),
                                     cov = cov)
        self.addLandmarks(landmarks)

    def addLandmarks(self,landmarks):
        self.landmarks.extend(landmarks)
        self.defineClassLabelsFromLandmarks()
        self.indexifyLandmarks()

    #goes over all landmarks to find classLabels.
    def defineClassLabelsFromLandmarks(self):
        if self.landmarks: #not empty
            classLabels = [lm.classLabel for lm in self.landmarks]
            classLabels = list(set(classLabels))
            if len(classLabels) > 10:
                raise Exception("no more than 10 classes are premited. Not enough distinguishable markers in matplotlib")
            else:
                self.classLabels = classLabels

    def indexifyLandmarks(self):
        for ii in range(len(self.landmarks)):
                self.landmarks[ii].index = ii

    def exportSemantics(self):
        semantics = {
                    "classLabel": self.classLabels,
                    "color": self.colors[:len(self.classLabels)],
                    "marker": self.markersList[:len(self.classLabels)]
                    }
        return semantics

    def plot(self,ax = None, plotIndex = False, plotCov = False,
                 markerSize = 10, markerColor = None, markerShape = None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()

        for lm in self.landmarks:
            ii = self.classLabels.index(lm.classLabel) #index of classLabel. used for shape and color

            if markerColor is None:
                markerColor = self.colors[ii].reshape(1,-1) #color from semantics

            if markerShape is None:
                markerShape = self.markersList[ii] #shape from semantics

            cov = None if plotCov is False else lm.cov
            index = None if plotIndex is False else lm.index

            utils.plotting.plot_landmark(ax, loc = lm.xy, cov = cov, 
                                index = index, 
                                markerColor = markerColor, 
                                markerShape = markerShape, 
                                markerSize = markerSize,
                                textColor = 'k')

    def transform(self,R,t):
        #transforms landmarks locations and convarance matrices
        # xy_new = R @ xy_old + t
        # cov_new = R @ cov_old @ R'
        for ii,lm in enumerate(self.landmarks):
            self.landmarks[ii].xy = np.squeeze(R @ lm.xy.reshape((2,1)) + t)
            self.landmarks[ii].cov = R @ lm.cov @ R.T

    def findClosestLandmark(self,otherMap):
        #returns 
        indices = []
        for lm in self.landmarks:
            indx = 0
            d = np.inf
            for lmOther in otherMap.landmarks:
                dTest = np.linalg.norm(lm.xy-lmOther.xy)
                if dTest < d:
                    indx = lmOther.index
                    d = dTest
            indices.append(indx)
        return indices

    def getLocations(self,indices = None):
        if indices is None:
            indices = range(len(self.landmarks))
        
        locations = []
        for ii in indices:
            locations.append(self.landmarks[ii].xy)
        return np.array(locations)

class landmark:
    def __init__(self, xy = None, classLabel = 'clutter', cov = None, index = None):
        
        if xy is None:
            xy = np.array([0,0])

        self.xy = xy
        self.classLabel = classLabel
        self.cov = cov
        self.index = index


    

    
