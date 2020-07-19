############################################################################
#
#   This is a Face Recognition model using Python face_recognision moddel
#   backed by dlib face recognition model.
#
#   --- This is For Storing Face Embedings   
#
############################################################################

import os
import pickle

import cv2

import face_recognition
from imutils import paths
import imutils
import sau


print("[INFO - imageToRegister_path] " + sau.imagesToRegister_path)

if(not os.path.exists(sau.registerData_path)):
    os.mkdir(sau.registerData_path)
print("[INFO - registerData_path] " + sau.registerData_path)

with os.scandir(sau.imagesToRegister_path) as rooms:
    for room in rooms:
        print("\t[INFO - Working on", room.name, "]")

        image_list= list(paths.list_images(room))
        if(len(image_list)> 0):

            persons_names= []
            persons_encodings= []

            for image in image_list:
                name= image.split("\\")[-2]
                #load_image= face_recognition.load_image_file(image)
                load_image= imutils.resize(
                    cv2.cvtColor(
                        cv2.imread(image),
                        cv2.COLOR_BGR2RGB
                    ),
                    width= 600
                )

                face_locations= face_recognition.face_locations(
                    load_image,
                    model="cnn",
                    number_of_times_to_upsample= 1
                )

                if(len(face_locations)>0):
                    face_encoding= face_recognition.face_encodings(
                        load_image, 
                        model="cnn", 
                        num_jitters= 1, 
                        known_face_locations=face_locations
                    )[0]

                    print(
                        "\t\t[INFO - ImageEmbeded] <Name=", name, ">",
                        "<No of Faces=", len(face_locations), ">",
                        "<location=", image, ">"
                    )

                    persons_encodings.append(face_encoding)
                    persons_names.append(name)

                else:
                    print(
                        "\t\t[INFO - FaceDetectFail] <Name=", name, ">",
                        "<location=", image, ">"
                    )

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