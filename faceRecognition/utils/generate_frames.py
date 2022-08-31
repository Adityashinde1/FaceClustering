import os
import cv2
import time
import shutil

# Extract 1 frame from each second from video footage
# and save the frames to a specific folder
def GenerateFrames(self, OutputDirectoryName):
    cap = cv2.VideoCapture(self.VideoFootageSource)
    _, frame = cap.read()
 
    fps = cap.get(cv2.CAP_PROP_FPS)
    TotalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
 
    print("[INFO] Total Frames ", TotalFrames, " @ ", fps, " fps")
    print("[INFO] Calculating number of frames per second")
 
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
 
    print('[INFO] Frames extracted')