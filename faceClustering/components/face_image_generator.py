import os
import sys
import time
import shutil
import numpy as np
import pickle
import logging
import cv2
from faceClustering.components.image_resize import ResizeUtils
from faceClustering.exception.exception import FaceClusteringCustomException
from imutils import build_montages

logger = logging.getLogger(__name__)

class FaceImageGenerator:
    def __init__(self, EncodingFilePath):
        self.EncodingFilePath = EncodingFilePath
 
    # Here we are creating montages for
    # first 25 faces for each distinct face.
    # We will also generate images for all
    # the distinct faces by using the labels
    # from clusters and image url from the
    # encodings pickle file.
     
    # The face bounding box is increased a
    # little more for training purposes and
    # we also created the exact annotation for
    # each face image (similar to darknet YOLO)
    # to easily adapt the annotation for future
    # use in supervised training
    def GenerateImages(self, labels, OutputFolderName = "ClusteredFaces", MontageOutputFolder = "Montage"):
        output_directory = os.getcwd()
 
        OutputFolder = os.path.join(output_directory, OutputFolderName)
        if not os.path.exists(OutputFolder):
            os.makedirs(OutputFolder)
        else:
            shutil.rmtree(OutputFolder)
            time.sleep(0.5)
            os.makedirs(OutputFolder)
 
        MontageFolderPath = os.path.join(OutputFolder, MontageOutputFolder)
        os.makedirs(MontageFolderPath)
 
        data = pickle.loads(open(self.EncodingFilePath, "rb").read())
        data = np.array(data)
 
        labelIDs = np.unique(labels)
         
        # loop over the unique face integers
        try:
            for labelID in labelIDs:
                # find all indexes into the `data` array
                # that belong to the current label ID, then
                # randomly sample a maximum of 25 indexes
                # from the set
                
                print("[INFO] faces for face ID: {}".format(labelID))
    
                FaceFolder = os.path.join(OutputFolder, "Face_" + str(labelID))
                os.makedirs(FaceFolder)
    
                idxs = np.where(labels == labelID)[0]
    
                # initialize the list of faces to
                # include in the montage
                portraits = []
    
                # loop over the sampled indexes
                counter = 1
                for i in idxs:
                    
                    # load the input image and extract the face ROI
                    image = cv2.imread(data[i]["imagePath"])
                    (o_top, o_right, o_bottom, o_left) = data[i]["loc"]
    
                    height, width, channel = image.shape
    
                    widthMargin = 100
                    heightMargin = 150
    
                    top = o_top - heightMargin
                    if top < 0: top = 0
                    
                    bottom = o_bottom + heightMargin
                    if bottom > height: bottom = height
                    
                    left = o_left - widthMargin
                    if left < 0: left = 0
                    
                    right = o_right + widthMargin
                    if right > width: right = width
    
                    portrait = image[top:bottom, left:right]
    
                    if len(portraits) < 25:
                        portraits.append(portrait)
    
                    resizeUtils = ResizeUtils()
                    portrait = resizeUtils.rescale_by_width(portrait, 400)
    
                    FaceFilename = "face_" + str(counter) + ".jpg"
    
                    FaceImagePath = os.path.join(FaceFolder, FaceFilename)
                    cv2.imwrite(FaceImagePath, portrait)
    
    
                    widthMargin = 20
                    heightMargin = 20
    
                    top = o_top - heightMargin
                    if top < 0: top = 0
                    
                    bottom = o_bottom + heightMargin
                    if bottom > height: bottom = height
                    
                    left = o_left - widthMargin
                    if left < 0: left = 0
                    
                    right = o_right + widthMargin
                    if right > width:
                        right = width
    
                    AnnotationFilename = "face_" + str(counter) + ".txt"
                    AnnotationFilePath = os.path.join(FaceFolder, AnnotationFilename)
                    
                    f = open(AnnotationFilePath, 'w')
                    f.write(str(labelID) + ' ' +
                            str(left) + ' ' + str(top) + ' ' +
                            str(right) + ' ' + str(bottom) + "\n")
                    f.close()
    
    
                    counter += 1
    
                montage = build_montages(portraits, (96, 120), (5, 5))[0]
                
                MontageFilenamePath = os.path.join(
                MontageFolderPath, "Face_" + str(labelID) + ".jpg")
                    
                cv2.imwrite(MontageFilenamePath, montage)

        except Exception as e:
            message = FaceClusteringCustomException(e, sys)
            logger.error(message.error_message)
            raise message.error_message