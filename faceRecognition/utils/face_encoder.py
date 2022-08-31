import cv2
import face_recognition
from pyPiper import Node


# Encode the face embedding, reference path
# and location and emit to pipeline
class FaceEncoder(Node):
    def setup(self, detection_method = 'cnn'):
        self.detection_method = detection_method
        # detection_method can be cnn or hog
 
    def run(self, data):
        id = data['id']
        imagePath = data['imagePath']
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        boxes = face_recognition.face_locations(
               rgb, model = self.detection_method)
 
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc}
                         for (box, enc) in zip(boxes, encodings)]
 
        self.emit({'id': id, 'encodings': d})