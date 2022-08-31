from image_resize import ResizeUtils

# The FramesGenerator extracts image
# frames  from the given video file
# The image frames are resized for
# face_recognition / dlib processing
class FramesGenerator:
    def __init__(self, VideoFootageSource):
        self.VideoFootageSource = VideoFootageSource
 
    # Resize the given input to fit in a specified
    # size for face embeddings extraction
    def AutoResize(self, frame):
        resizeUtils = ResizeUtils()
 
        height, width, _ = frame.shape
 
        if height > 500:
            frame = resizeUtils.rescale_by_height(frame, 500)
            self.AutoResize(frame)
         
        if width > 700:
            frame = resizeUtils.rescale_by_width(frame, 700)
            self.AutoResize(frame)
         
        return frame