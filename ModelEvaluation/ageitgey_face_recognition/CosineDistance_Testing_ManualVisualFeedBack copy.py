############################################################################
#
#   Testing the Recognition Manually Visual FeedBack
#
############################################################################

import os
import time
import pickle
import argparse
import numpy as np


from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_distances
import cv2
import dlib

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
    default=sau.load_image_width
)
ap.add_argument("-p", "--padding",
    help= "Provide image preprocessing resige width length is not needed because aspect ration will be kept the same",
    default=sau.padding
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

cnn_face_detector = dlib.cnn_face_detection_model_v1(sau.dlib_cnn_face_detection_model_path)
pose_predictor_68_point = dlib.shape_predictor(sau.dlib_68_shape_predictor_path)
dlib_facenet= dlib.face_recognition_model_v1(sau.dlib_facenet_model_path)

#################################################################
#   Loading The embedings from a room
#
StoreEmbedings= pickle.loads(open(embedings_path, "rb").read())
le= LabelEncoder()
encoded_lables= le.fit_transform(StoreEmbedings["names"])
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
        faces_bbs = cnn_face_detector(testImage, 1)
        no_of_faces= len(faces_bbs)

        if(no_of_faces>0):
            print("\t[INFO - DetectionSuccessful] <Detected Faces no=", no_of_faces, ">")

            print("\t[INFO - GeneratingFaceEmbedings]")
            faces_landmarks= dlib.full_object_detections()
            for face_bb in faces_bbs:
                faces_landmarks.append(pose_predictor_68_point(testImage, face_bb.rect))
            
            face_encodings= dlib_facenet.compute_face_descriptor(testImage, faces_landmarks, num_jitters=1, padding=float(args["padding"]))

            print("\t[INFO - GenerationSuccessful]")
            pred_distances, pred_idxs = ft.get_knn_cosine_distance(np.array(StoreEmbedings["encodings"]), np.array(face_encodings), k_neghbour=5)

            present= []
            for i, (pred_idx, pred_distance) in enumerate(zip(pred_idxs, pred_distances)):

                name, score= ft.get_knn_detection(pred_distance, pred_idx, le.classes_, encoded_lables)
                score *= 100
                

                # print("<", np.array([face_encoding]).shape, ">")
                #################################################################
                #   Marking The Presence of a Person
                #
                face_location=(faces_bbs[i].rect.top(), faces_bbs[i].rect.right(), faces_bbs[i].rect.bottom(), faces_bbs[i].rect.left())
                if(sau.markFaces(
                    testImage,
                    face_location, 
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