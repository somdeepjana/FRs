############################################################################
#
#   Testing the Recognition Manually Visual FeedBack
#
############################################################################

import os
import glob
import time
from datetime import datetime
import pickle
import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import cv2

import face_recognition
from imutils import paths
import imutils

import sau
import face_tools as ft

#################################################################
#   Defining all the commandline arguments
#
ap= argparse.ArgumentParser()
ap.add_argument("-c", "--capture",
    help="enter the capture source",
    default="0"
)
ap.add_argument("-r", "--room",
    help="Privide The Room To compare with",
    default="FriendsRoom"
)
ap.add_argument("-s", "--score",
    help= "Enter the Evaluation Score after which the prediction are positive",
    default=5
)
ap.add_argument("-m", "--model",
    help="Provide the dlib model for face detector",
    default="cnn",
    const="cnn",
    nargs='?',
    choices=["cnn", "hog"]
)
ap.add_argument("-e", "--embedding",
    help="Provide the dlib model for face embedding",
    default="large",
    const="large",
    nargs='?',
    choices=["large", "small"]
)
ap.add_argument("-u", "--upscale",
    help="Proivide the detection level upscale value",
    default=1
)
ap.add_argument("-j", "--jitters",
    help= "Provide the Recognization Level Jitters",
    default=1
)
ap.add_argument("-w", "--width",
    help= "Provide image preprocessing resige width length is not needed because aspect ration will be kept the same",
    default=sau.load_image_width
)
args= vars(ap.parse_args())
#
#################################################################

#################################################################
#   specifiying the paths
#
embedings_path= os.path.join(
    "RegisteredData",
    args["room"]+".room"
)
print("[INFO - FaceEmbedings] <path=", embedings_path, ">")

testVideo_path= os.path.join(
    "..",
    "ImageData",
    "TestVideo",
    args["capture"]
)
print("[INFO - TestVideo] <path=", testVideo_path, ">")
#
#################################################################

#################################################################
#   Loading The model and its clacess
#
StoreEmbedings= pickle.loads(open(embedings_path, "rb").read())
le= LabelEncoder()
encoded_lables= le.fit_transform(StoreEmbedings["names"])
#
#################################################################


try:
    testVideo_path= int(args["capture"])
except:
    testVideo_path= testVideo_path
testVideoStreme= cv2.VideoCapture(testVideo_path)
print("[INFO - VideoStremeLoaded] <Source=", testVideo_path, ">")

recording_start_time= datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%p")

present_names= []
present_scores= []
present_timestamps=[]
while(testVideoStreme.isOpened()):

    sucessRead, frame= testVideoStreme.read()

    if sucessRead:

        #   Preprocessing the Image befor pushing it to the CNN
        print("[INFO - PreprocessingFrame]")
        testFrame= imutils.resize(
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            ),
            width= int(args["width"])
        )
        print("[INFO - PreprocessingSuccessful]")

        # Detecting all the faces
        print("\t[INFO - DetectingFaces]")
        face_locations= face_recognition.face_locations(
            testFrame,
            model=args["model"],
            number_of_times_to_upsample= int(args["upscale"])
        )
        no_of_faces= len(face_locations)


        if(no_of_faces>0):
            print("\t[INFO - DetectionSuccessful] <Detected Faces no=", no_of_faces, ">")

            print("\t[INFO - GeneratingFaceEmbedings]")
            face_encodings= face_recognition.face_encodings(
                sau.pre_process_face(testFrame),
                model=args["embedding"],
                num_jitters= int(args["jitters"]),
                known_face_locations= face_locations
            )
            print("\t[INFO - GenerationSuccessful]")

            print("\t[INFO - PredictingFaces]")
            pred_distances, pred_idxs = ft.get_knn_cosine_distance(np.array(StoreEmbedings["encodings"]), np.array(face_encodings), k_neghbour=5)
            print("\t[INFO - PredictionSuccessful]")

            #################################################################
            #   Predicting The Presence of a Person
            #
            for i, (pred_idx, pred_distance) in enumerate(zip(pred_idxs, pred_distances)):
                name, score= ft.get_knn_detection(pred_distance, pred_idx, le.classes_, encoded_lables)
                score *= 100

                presence_idx= -1
                try:
                    presence_idx= present_names.index(name)
                except:
                    presence_idx= -1
                if(sau.markFaces(
                    testFrame,
                    face_locations[i], 
                    name, -score, -float(args["score"])
                )):
                    if(presence_idx == -1):
                        present_names.append(name)
                        present_scores.append(-score)
                        present_timestamps.append(str(datetime.now()))
                    else:
                        if(present_scores[presence_idx]>score):
                            present_scores[presence_idx]= -score
            #
            #################################################################
        cv2.imshow(
            "Video",
            cv2.cvtColor(
                testFrame,
                cv2.COLOR_RGB2BGR
            )
        )
        if cv2.waitKey(1) == ord('q'):
            break
        print("\t[INFO - PreviewingRecognitionComplete]")
    else:
        print("\t[INFO - FrameReadFail]")
        break

attendance_file= os.path.join(
    sau.attendance_path,
    recording_start_time+" to "+datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%p")
)

attendance_file= open(attendance_file+".csv", "w+")
attendance_file.write("Name,Time Stamp\n")
for present_name, present_timestamp in zip(present_names, present_timestamps):
    attendance_file.write(present_name+","+present_timestamp+"\n")
print("\t[INFO - AllPresents]\n", present_names, "\n", present_scores)
attendance_file.close()
testVideoStreme.release()
cv2.destroyAllWindows()