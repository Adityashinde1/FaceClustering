import os
import sys
import cv2
import time
import shutil
from faceClustering.components.image_resize import ResizeUtils
from faceClustering.exception.exception import FaceClusteringCustomException
import logging

logger = logging.getLogger(__name__)


# The FramesGenerator extracts image frames from the given video file. The image frames are resized for face_recognition / dlib processing.

class FramesGenerator:
    
    def __init__(self, VideoFootageSource):
        self.VideoFootageSource = VideoFootageSource
 
    # Resize the given input to fit in a specified
    # size for face embeddings extraction
    logger.info(f"AutoResize logging started...")
    
    def AutoResize(self, frame):
        try:

            
            resizeUtils = ResizeUtils()
    
            height, width, _ = frame.shape
    
            if height > 500:
                frame = resizeUtils.rescale_by_height(frame, 500)
                self.AutoResize(frame)
            
            if width > 700:
                frame = resizeUtils.rescale_by_width(frame, 700)
                self.AutoResize(frame)
            
            return frame

        except Exception as e:
            message = FaceClusteringCustomException(e, sys)
            logger.error(message.error_message)
            raise message.error_message

        # Extract 1 frame from each second from video footage
        # and save the frames to a specific folder
    def GenerateFrames(self, OutputDirectoryName):
            
        try:

            logger.info(f"Frames generating started...")
            cap = cv2.VideoCapture(self.VideoFootageSource)
            _, frame = cap.read()
        
            fps = cap.get(cv2.CAP_PROP_FPS)
            TotalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
            logger.info("Total Frames {} @ {} fps".format(TotalFrames, fps))
            logger.info(f"Calculating number of frames per second")
        
            CurrentDirectory = os.path.curdir
            OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectoryName)
        
            if os.path.exists(OutputDirectoryPath):
                shutil.rmtree(OutputDirectoryPath)
                time.sleep(0.5)
            os.mkdir(OutputDirectoryPath)
        
            CurrentFrame = 1
            fpsCounter = 0
            FrameWrittenCount = 1
            while CurrentFrame < TotalFrames:
                _, frame = cap.read()
                if (frame is None):
                    continue
                
                if fpsCounter > fps:
                    fpsCounter = 0
        
                    frame = self.AutoResize(frame)
        
                    filename = "frame_" + str(FrameWrittenCount) + ".jpg"
                    cv2.imwrite(os.path.join(OutputDirectoryPath, filename), frame)
        
                    FrameWrittenCount += 1
                
                fpsCounter += 1
                CurrentFrame += 1
        
            logger.info(f"Frames extracted...!")
        
        except Exception as e:
            message = FaceClusteringCustomException(e, sys)
            logger.error(message.error_message)
            raise message.error_message