import os

from PIL import Image, ImageDraw
import time
import cv2

from mtcnn import MTCNN

testimage_path= os.path.join(
    "ImageData",
    "TestImages",
    "Random",
    "faces.jpg"
)

img = cv2.cvtColor(cv2.imread(testimage_path), cv2.COLOR_BGR2RGB)

detector= MTCNN()

start_time= time.time()
detected_boxes= detector.detect_faces(img)

print("Detection and Landmark Completed in: ", time.time()-start_time)