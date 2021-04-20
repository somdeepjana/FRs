############################################################################
#
#   Testing the Recognition Manually Visual FeedBack
#
############################################################################

import os
import glob
import pickle
import argparse
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from sklearn.calibration import CalibratedClassifierCV
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
    default=0.3
)
# ap.add_argument("-d", "--thresold_distance",
#     help= "Enter the thresold distance, any prediction exceeding this distance will be concidered as zero",
#     default=3
# )
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
#   Loading The model and its clacess
#
recognizerModel= pickle.loads(open(trainedModel_path, "rb").read())
#
#################################################################

testImage_paths= list(paths.list_images(testFolder_path))
no_of_testImages= len(testImage_paths)
print("[INFO - TotalTestImages] <No of Test Images=", no_of_testImages, ">")

if(no_of_testImages>0):
    for (i_img, testImage_path) in enumerate(testImage_paths):
        testImage_name= testImage_path.split("\\")[-1]

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

            print("\t[INFO - PredictingFaces]")
            # predictions= recognizerModel.predict(face_encodings)
            # predictions= recognizerModel.predict(normalize(np.array(face_encodings)))
            # pred_distances= recognizerModel.kneighbors(face_encodings)
            pred_distances, pred_idxs= recognizerModel.kneighbors(normalize(np.array(face_encodings)))
            print(pred_distances.shape)
            print("\t[INFO - PredictionSuccessful]")

            #################################################################
            #   Predicting The Presence of a Person
            #
            present= []
            for i, (pred_idx, pred_distance) in enumerate(zip(pred_idxs, pred_distances)):
                name, score= ft.get_knn_detection(pred_distance, pred_idx, recognizerModel.classes_, recognizerModel._y)

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

            print("\t[INFO - PreviewingRecognition]")
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