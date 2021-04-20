############################################################################
#
#   Testing the Recognition Manually Visual FeedBack
#
############################################################################

import os
import glob
import time
import pickle
import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
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
    default=0.3
)
ap.add_argument("-m", "--model",
    help="Provide the dlib model for face detector",
    default="cnn",
    const="cnn",
    nargs='?',
    choices=["cnn", "hog"]
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
trainedModel_path= os.path.join(
    "trainedSVM",
    args["room"]+".knn"
)
print("[INFO - TrainedModel] <path=", trainedModel_path, ">")

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
recognizerModel= pickle.loads(open(trainedModel_path, "rb").read())
#
#################################################################


try:
    testVideo_path= int(args["capture"])
except:
    testVideo_path= testVideo_path
testVideoStreme= cv2.VideoCapture(testVideo_path)
print("[INFO - VideoStremeLoaded] <Source=", testVideo_path, ">")

present_name= []
present_score= []
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
                model="large",
                num_jitters= int(args["jitters"]),
                known_face_locations= face_locations
            )
            print("\t[INFO - GenerationSuccessful]")

            print("\t[INFO - PredictingFaces]")
            pred_distances, pred_idxs= recognizerModel.kneighbors(normalize(np.array(face_encodings)))
            print("\t[INFO - PredictionSuccessful]")

            #################################################################
            #   Predicting The Presence of a Person
            #
            for i, (pred_idx, pred_distance) in enumerate(zip(pred_idxs, pred_distances)):
                name, score= ft.get_knn_detection(pred_distance, pred_idx, recognizerModel.classes_, recognizerModel._y)

                presence_idx= -1
                try:
                    presence_idx= present_name.index(name)
                except:
                    presence_idx= -1
                if(sau.markFaces(
                    testFrame,
                    face_locations[i], 
                    name, -score, -float(args["score"])
                )):
                    if(presence_idx == -1):
                        present_name.append(name)
                        present_score.append(-score)
                    else:
                        if(present_score[presence_idx]>score):
                            present_score[presence_idx]= -score
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
    
print("\t[INFO - AllPresents]\n", present_name, "\n", present_score)
testVideoStreme.release()
cv2.destroyAllWindows()