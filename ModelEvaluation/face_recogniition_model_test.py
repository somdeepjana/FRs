################ dcumentation help
# https://face-recognition.readthedocs.io/en/latest/face_recognition.html

################ Implimentation Help
# https://beta.deepnote.com/project/1511fb1d-4aa8-4d67-a79f-de624c99b0d2#%2Fexample.ipynb

import os

from PIL import Image, ImageDraw
import time

import face_recognition

testimage_path= os.path.join(
    "ImageData",
    "TestImages",
    "PyImages",
    "adrian",
    "00005.jpg"
)

strt_time= time.time()
img= face_recognition.load_image_file("00005.jpg")
print("Load Image File Type: ", type(img), "Image  Loading Time: ", time.time()-strt_time)

strt_time= time.time()
face_locations= face_recognition.face_locations(img, model="cnn", number_of_times_to_upsample=1)
print("\nFace Location Returns: ", type(face_locations), "No of Fcaes: ", len(face_locations), "Finding Locations Time: ", time.time()-strt_time)
print("A peek in retrive locations: ", face_locations[0:3])

strt_time= time.time()
face_landmarks= face_recognition.face_landmarks(img, face_locations=face_locations)
print("\nFace landmark Returns: ", type(face_landmarks), "No of Fcaes: ", len(face_landmarks), "Finding Landmarks Time: ", time.time()-strt_time)
print("A peek in retrive landmark: ", face_landmarks[0].keys())

strt_time= time.time()
face_encoings= face_recognition.face_encodings(img, model="cnn", known_face_locations=face_locations)
print("\nFace Encodings Returns: ", type(face_encoings), "No of Fcaes: ", len(face_encoings), "Finding Landmarks Time: ", time.time()-strt_time)
print("A Peek in retrive Data: ", face_encoings[0].shape)

pil_image= Image.fromarray(img)

draw= ImageDraw.Draw(pil_image)

for face in face_locations:
    (top, right, bottome, left)= face

    draw.rectangle(((left, top), (right, bottome)), outline=(255, 0, 0), width=3)
    '''
    coordinates = [(x1, y1), (x2, y2)]

        (x1, y1)
            *--------------
            |             |
            |             |
            |             |
            |             |
            |             |
            |             |
            --------------*
                        (x2, y2)
                        
    https://stackoverflow.com/questions/34255938/is-there-a-way-to-specify-the-width-of-a-rectangle-in-pil
    '''

pil_image.show()