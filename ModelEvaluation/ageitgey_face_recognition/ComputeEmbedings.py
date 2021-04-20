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
from sklearn.neighbors import LocalOutlierFactor

import face_recognition
import numpy as np
from imutils import paths
import imutils
import sau


print("[INFO - imageToRegister_path] " + sau.imagesToRegister_path)

if(not os.path.exists(sau.registerData_path)):
    os.mkdir(sau.registerData_path)
print("[INFO - registerData_path] " + sau.registerData_path)

with os.scandir(sau.imagesToRegister_path) as rooms:
    for room in rooms:
        # print("\t[INFO - Working on", room.name, "]")

        image_list= list(paths.list_images(room))
        if(len(image_list)> 0):

            persons_names= []
            persons_encodings= []

            image_batch= []
            t_image_list= tqdm(image_list, desc= "[INFO - Working on " + room.name + " ]")
            for image in t_image_list:
                t_image_list.set_postfix_str(image.split("\\")[-1])

                name= image.split("\\")[-2]
                #load_image= face_recognition.load_image_file(image)
                load_image= imutils.resize(
                    cv2.cvtColor(
                        cv2.imread(image),
                        cv2.COLOR_BGR2RGB
                    ),
                    width= sau.load_image_width
                )

                face_locations= face_recognition.face_locations(
                    load_image,
                    model="cnn",
                    number_of_times_to_upsample= 1
                )

                if(len(face_locations)>0):
                    face_encoding= face_recognition.face_encodings(
                        sau.pre_process_face(load_image), 
                        model="large", 
                        num_jitters= 1, 
                        known_face_locations=face_locations
                    )[0]

                    persons_encodings.append(face_encoding)
                    persons_names.append(name)

                else:
                    print(
                        "\t\t[INFO - FaceDetectFail] <Name=", name, ">",
                        "<location=", image, ">"
                    )

            t_image_list.set_postfix_str("Complete")

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

