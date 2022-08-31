import os
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# Face clustering functionality
class FaceClusterUtility:
     
    def __init__(self, EncodingFilePath):
        self.EncodingFilePath = EncodingFilePath
     
    # Credits: Arian's pyimagesearch for the clustering code
    # Here we are using the sklearn.DBSCAN functionality
    # cluster all the facial embeddings to get clusters
    # representing distinct people
    def Cluster(self):
        InputEncodingFile = self.EncodingFilePath
        if not (os.path.isfile(InputEncodingFile) and
                os.access(InputEncodingFile, os.R_OK)):
            print('The input encoding file, ' +
                    str(InputEncodingFile) +
                    ' does not exists or unreadable')
            exit()
 
        NumberOfParallelJobs = -1
 
        # load the serialized face encodings
        # + bounding box locations from disk,
        # then extract the set of encodings to
        # so we can cluster on them
        print("[INFO] Loading encodings")
        data = pickle.loads(open(InputEncodingFile, "rb").read())
        data = np.array(data)
 
        encodings = [d["encoding"] for d in data]
 
        # cluster the embeddings
        print("[INFO] Clustering")
        clt = DBSCAN(eps = 0.5, metric ="euclidean",
                      n_jobs = NumberOfParallelJobs)
                       
        clt.fit(encodings)
 
        # determine the total number of
        # unique faces found in the dataset
        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("[INFO] # unique faces: {}".format(numUniqueFaces))
 
        return clt.labels_



# Inherit class tqdm for visualization of progress
class TqdmUpdate(tqdm):
     
    # This function will be passed as progress
    # callback function. Setting the predefined
    # variables for auto-updates in visualization
    def update(self, done, total_size = None):
        if total_size is not None:
            self.total = total_size
             
        self.n = done
        super().refresh()