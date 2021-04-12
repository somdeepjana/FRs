import os
import argparse

import numpy as np
import cv2

import dlib
from PIL import Image

import imutils
import sau


ap= argparse.ArgumentParser()
ap.add_argument("-n", "--name",
    help="Name of the person.",
    default="test",
    # required=True
)
ap.add_argument("-r", "--room",
    help="Name of the Room to store the faces in.",
    default="FriendsRoom",
    # required=True
)
ap.add_argument("-c", "--capture",
    help="Video feed to captue from",
    default="0",
    # required=True
)
args= vars(ap.parse_args())

store_directory= os.path.join(
    sau.imagesToRegister_path,
    args["room"],
    args["name"]
)

if not os.path.exists(store_directory):
    os.makedirs(store_directory)

try:
    captureVideo_path= int(args["capture"])
except:
    captureVideo_path= testVideo_path


cnn_face_detector = dlib.cnn_face_detection_model_v1(sau.dlib_cnn_face_detection_model_path)
pose_predictor_68_point = dlib.shape_predictor(sau.dlib_68_shape_predictor_path)

cap= cv2.VideoCapture(captureVideo_path)

frame_no= 0
face_capture_no= 0
while(cap.isOpened()):
    ret, frame= cap.read()
    if not ret:
        print("\t[INFO - FrameReadFail]")
        break

    print("[INFO - PreprocessingFrame] <No ", frame_no, " >")
    captureFrame= imutils.resize(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            width= 700
        )
    print("[INFO - PreprocessingSuccessful]")
    # Detecting all the faces
    print("\t[INFO - DetectingFaces]")
    detections = cnn_face_detector(captureFrame, 1)
    # print(detections)
    # exit()
    no_detections= len(detections)
    if(no_detections >1):
        print("\t[WARN - MultipleFace] < Not saving any face from frame No ", frame_no, " >")

    elif(no_detections==0):
        print("\t[WARN - NoFace] < Not saving any face from frame No ", frame_no, " >")

    else:
        face_landmarks= pose_predictor_68_point(captureFrame, detections[0].rect)
        captureFrame= dlib.get_face_chip(captureFrame, face_landmarks, size=320)
        cv2.imwrite(
            os.path.join(
                store_directory,
                str(face_capture_no)+".jpg"
            ),
            cv2.cvtColor(
                captureFrame,
                cv2.COLOR_RGB2BGR
            ) 
        )

        face_capture_no += 1
        if(face_capture_no >100):
            break

    cv2.imshow(
        "Recording Face of "+ args["name"],
        cv2.cvtColor(
            captureFrame,
            cv2.COLOR_RGB2BGR
        )    
    )

    frame_no += 1
    if cv2.waitKey(1) == ord('q'):
        break


print("\t[INFO - RecordingComplete]")

cap.release()
cv2.destroyAllWindows()