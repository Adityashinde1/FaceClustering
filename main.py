import os
from faceRecognition.utils.frames_generator import FramesGenerator
from faceRecognition.utils.datastore_manager import DatastoreManager
from faceRecognition.utils.face_cluster_utility import TqdmUpdate, FaceClusterUtility
from faceRecognition.utils.face_encoder import FaceEncoder
from faceRecognition.utils.Frames_provider import FramesProvider
from faceRecognition.utils.pickle_listcollector import PicklesListCollator
from faceRecognition.utils.face_image_generator import FaceImageGenerator
import logging
import shutil
import time
from pyPiper import Node, Pipeline

logging_level = logging.INFO
main_logger = logging.getLogger()
main_logger.setLevel(logging_level)

# Set up a stream handler to log to the console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging_level)
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

# Add handler to logger
main_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
     
    try:
    # Generate the frames from given video footage
        framesGenerator = FramesGenerator("Footage.mp4")
        framesGenerator.GenerateFrames("Frames")

    except Exception as e:
        print(e)


    ''' Design and run the face clustering pipeline '''

    CurrentPath = os.getcwd()
    FramesDirectory = "Frames"
    FramesDirectoryPath = os.path.join(CurrentPath, FramesDirectory)
    EncodingsFolder = "Encodings"
    EncodingsFolderPath = os.path.join(CurrentPath, EncodingsFolder)

    try:    
        if os.path.exists(EncodingsFolderPath):
            shutil.rmtree(EncodingsFolderPath, ignore_errors=True)
            time.sleep(0.5)
        os.makedirs(EncodingsFolderPath)

    except Exception as e:
        print(e)



    pipeline = Pipeline(
                        FramesProvider("Files source", sourcePath=FramesDirectoryPath) | 
                        FaceEncoder("Encode faces") | 
                        DatastoreManager("Store encoding", 
                        encodingsOutputPath=EncodingsFolderPath), 
                        n_threads = 3, quiet = True)
    pbar = TqdmUpdate()
    pipeline.run(update_callback=pbar.update)

    logger.info(f"Encodings Extracted..")

    ''' Merge all the encodings pickle files into one '''
    CurrentPath = os.getcwd()
    EncodingsInputDirectory = "Encodings"
    EncodingsInputDirectoryPath = os.path.join(CurrentPath, EncodingsInputDirectory)

    OutputEncodingPickleFilename = "encodings.pickle"

    try:
        if os.path.exists(OutputEncodingPickleFilename):
            os.remove(OutputEncodingPickleFilename)

    except Exception as e:
        print(e)

    picklesListCollator = PicklesListCollator(EncodingsInputDirectoryPath)
    picklesListCollator.GeneratePickle(OutputEncodingPickleFilename)

    # To manage any delay in file writing
    time.sleep(0.5)

    logger.info(f"Start clustering process and generate output images with annotations")
    EncodingPickleFilePath = "encodings.pickle"

    faceClusterUtility = FaceClusterUtility(EncodingPickleFilePath)
    faceImageGenerator = FaceImageGenerator(EncodingPickleFilePath)

    labelIDs = faceClusterUtility.Cluster()
    faceImageGenerator.GenerateImages(labelIDs, "ClusteredFaces", "Montage")