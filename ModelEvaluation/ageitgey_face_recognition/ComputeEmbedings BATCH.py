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
import math
from tqdm import tqdm

import cv2
import dlib
from sklearn.neighbors import LocalOutlierFactor

import face_recognition
import numpy as np
from imutils import paths
import imutils
import sau
import face_tools as cb


dlib_facenet= dlib.face_recognition_model_v1(sau.dlib_facenet_model_path)

print("[INFO - imageToRegister_path] " + sau.imagesToRegister_path)

if(not os.path.exists(sau.registerData_path)):
    os.mkdir(sau.registerData_path)
print("[INFO - registerData_path] " + sau.registerData_path)

with os.scandir(sau.imagesToRegister_path) as rooms:
    for room in rooms:
        # print("\t[INFO - Working on", room.name, "]")

        image_list= list(paths.list_images(room))
        no_img_list= len(image_list)
        if(no_img_list> 0):

            persons_names= []
            persons_encodings= []

            for batch in sau.batch_retrive(image_list, sau.batch_size):
                img_batch, lable_batch= cb.load_batch_images_from_list(batch, width=150, square=True, pre_process=True)
                faces_embeddings= dlib_facenet.compute_face_descriptor(img_batch, num_jitters=1)
                persons_names.extend(lable_batch)
                persons_encodings.extend(faces_embeddings)

            # clf = LocalOutlierFactor(n_neighbors=6)
            # outlier_idx= clf.fit_predict(persons_encodings)

            # persons_encodings= list(np.array(persons_encodings)[outlier_idx==1])
            # persons_names = list(np.array(persons_names)[outlier_idx==1])

            np.savetxt("./analyze/"+room.name+"_vecs.tsv", persons_encodings, delimiter='\t')
            out_m = io.open("./analyze/"+room.name+'_meta.tsv', 'w', encoding='utf-8')
            for labels in persons_names:
                out_m.write(labels + "\n")
            out_m.close()



            room_serialized={
                "names": persons_names,
                "encodings": persons_encodings
            }

            serialize_file= open(os.path.join(
                sau.registerData_path,
                room.name + ".room"
            ), "wb")
            serialize_file.write(pickle.dumps(room_serialized))
            serialize_file.close()
        else:
            print("\t[INFO - NoImagePresent]")

