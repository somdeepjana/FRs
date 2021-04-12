
import sys

import dlib

import cv2

predictor_path = "C:\\Users\\somdeep\\.conda\\envs\\FRs\\Lib\\site-packages\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat"
face_file_path = "C:/Users/somdeep/Documents/ComputerVisionProjects/FRs/ModelEvaluation/ImageData/TestImages/Nitesh_Test/me_in_test1.jpg"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

# Load the image using Dlib
img = dlib.load_rgb_image(face_file_path)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.
dets = detector(img, 1)
# print(dets.shape)

num_faces = len(dets)
if num_faces == 0:
    print("Sorry, there were no faces found in '{}'".format(face_file_path))
    exit()

# Find the 5 face landmarks we need to do the alignment.
faces = dlib.full_object_detections()
for detection in dets:
    faces.append(sp(img, detection))

# Get the aligned face images
# Optionally: 
# images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
images = dlib.get_face_chips([img,img], [faces,faces], size=320, padding=0.25)
# print(images)
for image in images:

    for i in image:
        cv2.imshow("",cv2.cvtColor(i, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)

