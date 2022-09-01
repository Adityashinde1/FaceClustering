import os
import sys
from faceClustering.components.frames_generator import FramesGenerator
from faceClustering.components.datastore_manager import DatastoreManager
from faceClustering.components.face_cluster_utility import TqdmUpdate, FaceClusterUtility
from faceClustering.components.face_encoder import FaceEncoder
from faceClustering.components.frames_provider import FramesProvider
from faceClustering.components.pickle_listcollector import PicklesListCollator
from faceClustering.components.face_image_generator import FaceImageGenerator
from faceClustering.exception.exception import FaceClusteringCustomException
import logging
import shutil
import time
from pyPiper import Pipeline

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # creating path
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
        message = FaceClusteringCustomException(e, sys)
        logger.error(message.error_message)
        raise message.error_message

    
 
    try:
    # Generate the frames from given video footage
        framesGenerator = FramesGenerator(r"F:\live_face_recognition\live_face_recognition\faceClustering\data\Footage.mp4")
        framesGenerator.GenerateFrames(FramesDirectoryPath)

    except Exception as e:
        message = FaceClusteringCustomException(e, sys)
        logger.error(message.error_message)
        raise message.error_message

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
        message = FaceClusteringCustomException(e, sys)
        logger.error(message.error_message)
        raise message.error_message

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