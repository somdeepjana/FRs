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


from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_distances
import cv2

import face_recognition
from imutils import paths
import imutils

import sau

#################################################################
#   Defining all the commandline arguments
#
ap= argparse.ArgumentParser()
ap.add_argument("-t", "--testfolder",
    help="Provide The folder Name of TestImages",
    default="Tarapada_Test"
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
    default=700
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

testFolder_path= os.path.join(
    "..",
    "ImageData",
    "TestImages",
    args["testfolder"]
)
print("[INFO - TestImages] <path=", testFolder_path, ">")
#
#################################################################

#################################################################
#   Loading The embedings from a room
#
StoreEmbedings= pickle.loads(open(embedings_path, "rb").read())
#
#################################################################

testImage_paths= list(paths.list_images(testFolder_path))
no_of_testImages= len(testImage_paths)
print("[INFO - TotalTestImages] <No of Test Images=", no_of_testImages, ">")

if(no_of_testImages>0):
    for (i_img, testImage_path) in enumerate(testImage_paths):
        testImage_name= testImage_path.split("\\")[-1]

        start_time= time.time()

        #   Preprocessing the Image befor pushing it to the CNN
        print("[INFO - LoadingPreprocess] <Image=", testImage_name, ">")
        testImage= imutils.resize(
            cv2.cvtColor(
                cv2.imread(testImage_path),
                cv2.COLOR_BGR2RGB
            ),
            width= int(args["width"])
        )
        print("[INFO - LoadedPreprocessedSuccessful]")

        # Detecting all the faces
        print("\t[INFO - DetectingFaces]")
        face_locations= face_recognition.face_locations(
            testImage,
            model=args["model"],
            number_of_times_to_upsample= int(args["upscale"])
        )
        no_of_faces= len(face_locations)

        if(no_of_faces>0):
            print("\t[INFO - DetectionSuccessful] <Detected Faces no=", no_of_faces, ">")

            print("\t[INFO - GeneratingFaceEmbedings]")
            face_encodings= face_recognition.face_encodings(
                sau.pre_process_face(testImage),
                model="large",
                num_jitters= int(args["jitters"]),
                known_face_locations= face_locations
            )
            print("\t[INFO - GenerationSuccessful]")

            present= []
            for (i, face_encoding) in enumerate(face_encodings):

                print("\t[INFO - PredictingFaces]")
                distances= cosine_distances(np.array(StoreEmbedings["encodings"]), np.array([face_encoding]))
                distances= np.squeeze(distances)
                print("\t[INFO - PredictionSuccessful]")

                nearest= np.argmin(distances)
                score= distances[nearest] * 100
                name= StoreEmbedings["names"][nearest]

                # print("<", np.array([face_encoding]).shape, ">")
                #################################################################
                #   Marking The Presence of a Person
                #
                if(sau.markFaces(
                    testImage,
                    face_locations[i], 
                    name, -score, -float(args["score"])
                )):
                    present.append({
                        "name": name,
                        "Score": -score
                    })
                #
                #################################################################

            print("\t[INFO - PreviewingRecognition]", time.time()-start_time)
            print("\t[INFO - Presented]", present)
            cv2.imshow(
                testImage_name,
                cv2.cvtColor(
                    testImage,
                    cv2.COLOR_RGB2BGR
                )
            )
            if cv2.waitKey(0) == ord('q'):
                break
            print("\t[INFO - PreviewingRecognitionComplete]")
        else:
            print("\t[INFO - DetectionFailed] <Path=", testImage_path, ">")