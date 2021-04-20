############################################################################
#
#   This is a Face Recognition model using Python face_recognision moddel
#   backed by dlib face recognition model.
#
#   --- This is For Storing Face Embedings   
#
############################################################################

import os
import io
import pickle
from tqdm import tqdm

import cv2
import dlib
from sklearn.neighbors import LocalOutlierFactor

import face_recognition
import numpy as np
from imutils import paths
import imutils
import sau

cnn_face_detector = dlib.cnn_face_detection_model_v1(sau.dlib_cnn_face_detection_model_path)
pose_predictor_68_point = dlib.shape_predictor(sau.dlib_68_shape_predictor_path)


print("[INFO - imageToRegister_path] " + sau.imagesToRegister_path)

with os.scandir(sau.rawImage_path) as rooms:
    for room in rooms:
        # print("\t[INFO - Working on", room.name, "]")

        image_list= list(paths.list_images(room))
        if(len(image_list)> 0):
            t_image_list= tqdm(image_list, desc= "[INFO - Working on " + room.name + " ]")

            for image in t_image_list:
                file_name= image.split("\\")[-1]
                name= image.split("\\")[-2]

                t_image_list.set_postfix_str(file_name)

                processed_store_directory= os.path.join(
                    sau.imagesToRegister_path,
                    room.name,
                    name
                )
                if not os.path.exists(processed_store_directory):
                    os.makedirs(processed_store_directory)

                load_image= imutils.resize(
                    cv2.cvtColor(
                        cv2.imread(image),
                        cv2.COLOR_BGR2RGB
                    ),
                    width= sau.load_image_width
                )

                detections = cnn_face_detector(load_image, 1)
                no_detections= len(detections)

                if(no_detections >1):
                    print("\t[WARN - MultipleFace] < Not saving any face from ", file_name, " >")
                elif(no_detections==0):
                    print("\t[WARN - NoFace] < Not saving any face from ", file_name, " >")
                else:
                    face_landmarks= pose_predictor_68_point(load_image, detections[0].rect)
                    load_image= dlib.get_face_chip(load_image, face_landmarks, size=sau.processed_face_size, padding=sau.padding)
                    cv2.imwrite(
                        os.path.join(
                            processed_store_directory,
                            file_name
                        ),
                        cv2.cvtColor(
                            load_image,
                            cv2.COLOR_RGB2BGR
                        ) 
                    )

            t_image_list.set_postfix_str("Complete")

        else:
            print("\t[INFO - NoImagePresent]")

