import os
import pickle

# PicklesListCollator takes multiple pickle
# files as input and merges them together
# It is made specifically to support use-case
# of merging distinct pickle files into one
class PicklesListCollator:
    def __init__(self, picklesInputDirectory):
        self.picklesInputDirectory = picklesInputDirectory
     
    # Here we will list down all the pickles
    # files generated from multiple threads,
    # read the list of results append them to a
    # common list and create another pickle
    # with combined list as content
    def GeneratePickle(self, outputFilepath):
        datastore = []
 
        ListOfPickleFiles = []
        for item in os.listdir(self.picklesInputDirectory):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.pickle':
                ListOfPickleFiles.append(os.path.join(
                    self.picklesInputDirectory, item))
 
        for picklePath in ListOfPickleFiles:
            with open(picklePath, "rb") as f:
                data = pickle.loads(f.read())
                datastore.extend(data)
 
        with open(outputFilepath, 'wb') as f:
            f.write(pickle.dumps(datastore))