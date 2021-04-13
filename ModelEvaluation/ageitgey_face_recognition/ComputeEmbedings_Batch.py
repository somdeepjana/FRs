
import time
import numpy as np
import cv2

import dlib

import imutils

import sau


cnn_face_detector = dlib.cnn_face_detection_model_v1(sau.dlib_cnn_face_detection_model_path)
pose_predictor_68_point = dlib.shape_predictor(sau.dlib_68_shape_predictor_path)
dlib_facenet= dlib.face_recognition_model_v1(sau.dlib_facenet_model_path)

def get_batch_embedding(batch_images):
    face_detect_batch= cnn_face_detector(batch_images, 1)

    face_lanmark_batch= []
    for img_no ,img_rec in enumerate(face_detect_batch):
        face_landmarks= dlib.full_object_detections() # full_object_detections in each image
        for face_bb in img_rec:
            face_landmarks.append(pose_predictor_68_point(batch_images[img_no], face_bb.rect)) # predicting each face landmark in a Image of given Batch Image
        
        face_lanmark_batch.append(face_landmarks)
    return dlib_facenet.compute_face_descriptor(batch_images, face_lanmark_batch, num_jitters=1), face_detect_batch

    
def get_batch_crop_faces_from_diff_input_size(batch_images):

    aligned_faces= []
    batch_landmarks=[]
    img_face_list= []

    for img in batch_images:
        faces_bbs= cnn_face_detector(img, 1)
        faces_landmarks= dlib.full_object_detections()
        no_of_facec_in_img= 0
        for face_bb in faces_bbs:
            faces_landmarks.append(pose_predictor_68_point(img, face_bb.rect))
            no_of_facec_in_img += 1
        aligned_faces.extend(dlib.get_face_chips(img, faces_landmarks, size=150))
        batch_landmarks.append(faces_landmarks)
        img_face_list.append(no_of_facec_in_img)

    return aligned_faces, batch_landmarks, img_face_list




if __name__ == "__main__":
    img1= cv2.imread("C:\\Users\\somdeep\\Documents\\ComputerVisionProjects\\FRs\\ModelEvaluation\\ImageData\\TestImages\\Tarapada_Test\\20191112131430_IMG_1561.JPG.jpg", cv2.COLOR_BGR2RGB)
    img1= cv2.resize(img1, (825, 700))
    
    img2= cv2.imread("C:\\Users\\somdeep\\Documents\\ComputerVisionProjects\\FRs\\ModelEvaluation\\ImageData\\TestImages\\Tarapada_Test\\IMG_20190708_122350.jpg", cv2.COLOR_BGR2RGB)
    img2= cv2.resize(img2, (825, 700))

    # img_batch= [img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2]
    img_batch= [img1, img2]

    # img_batch= cv2.dnn.blobFromImages(img_batch, size= (700,825))
    # print(img_batch.shape)

    ########### batch_embedding
    # batch_embeddings, batch_landmark= get_batch_embedding(img_batch)
    # print(len(batch_embeddings), len(batch_landmark))
    # for facess_emb, faces_landmarks, img in zip(batch_embeddings, batch_landmark, img_batch):
    #     for face_embedding, face_landmark in zip(facess_emb, faces_landmarks):
    #         bb= face_landmark.rect
    #         sau.markFaces(img, [bb.top(), bb.right(), bb.bottom(), bb.left()], "name", 90, 80)
    #     cv2.imshow("", img)
    #     cv2.waitKey(0)

    ######## batch crop
    # start_time= time.time()
    # print(len(get_batch_crop_faces_from_diff_input_size(img_batch)), time.time()-start_time)

    # batch_aling, batch_landmar= get_batch_crop_faces_from_diff_input_size(img_batch)
    # for img, bbs in zip(img_batch, batch_landmar):
    #     for bb in bbs:
    #         # print(bb.num_parts)
    #         sau.markFaces(img, [bb.rect.top(), bb.rect.right(), bb.rect.bottom(), bb.rect.left()], "name", 90, 80)
    #     cv2.imshow("", img)
    #     cv2.waitKey(0)