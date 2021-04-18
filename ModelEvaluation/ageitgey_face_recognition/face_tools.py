
import math

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
    face_detect_batch= cnn_face_detector(batch_images, batch_size=3)

    face_lanmark_batch= []
    for img_no ,img_rec in enumerate(face_detect_batch):
        face_landmarks= dlib.full_object_detections() # full_object_detections in each image
        for face_bb in img_rec:
            face_landmarks.append(pose_predictor_68_point(batch_images[img_no], face_bb.rect)) # predicting each face landmark in a Image of given Batch Image
        
        face_lanmark_batch.append(face_landmarks)
    return dlib_facenet.compute_face_descriptor(batch_images, face_lanmark_batch, num_jitters=1), face_detect_batch

    
def get_batch_crop_faces_from_diff_input_size(batch_images, lables, slient_drop_multiface= False):

    aligned_faces= []
    detect_lables= []
    batch_landmarks=[]
    img_face_list= []

    for img, label in zip(batch_images, lables):
        no_of_faces_in_img, landmarks_in_img, align_faces_in_img = get_crop_face_from_image(img)
        if(slient_drop_multiface and no_of_faces_in_img>1):
            continue
        aligned_faces.extend(align_faces_in_img)
        detect_lables.append(label)
        batch_landmarks.append(landmarks_in_img)
        img_face_list.append(no_of_faces_in_img)

    return aligned_faces, detect_lables, batch_landmarks, img_face_list


def get_crop_face_from_image(img):
    faces_bbs= cnn_face_detector(img, 1)
    faces_landmarks= dlib.full_object_detections()
    no_of_facec_in_img= 0
    for face_bb in faces_bbs:
        faces_landmarks.append(pose_predictor_68_point(img, face_bb.rect))
        no_of_facec_in_img += 1
        # print(img.shape)
        # exit()
    return no_of_facec_in_img, faces_landmarks, dlib.get_face_chips(img, faces_landmarks, size=150)


def load_batch_images_from_list(img_paths, width=700):

    batch_images=[]
    batch_lables=[]

    for img_path in img_paths:
        temp_img= imutils.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), width=width)
        batch_images.append(temp_img)
        batch_lables.append(img_path.split('\\')[-2])
    return batch_images, batch_lables


def get_knn_detection(pred_distance, pred_idx, total_classes, recognizer_y):
    nearest_k_classes_no= recognizer_y[pred_idx] # Index of nearest K classes
    highest_frequency_class_no= np.argmax(np.bincount(nearest_k_classes_no)) # index of the class with highest frequency
    idx_of_highest_frequency= np.where(nearest_k_classes_no==highest_frequency_class_no) # indexs of the highest frequency pred class in nearest_k_classes_no

    name= total_classes[highest_frequency_class_no]

    score= np.amin(pred_distance[idx_of_highest_frequency])

    return name, score


if __name__ == "__main__":
    img1_path="C:\\Users\\somdeep\\Documents\\ComputerVisionProjects\\FRs\\ModelEvaluation\\ImageData\\TestImages\\Tarapada_Test\\20191112131430_IMG_1561.JPG.jpg"
    img2_path="C:\\Users\\somdeep\\Documents\\ComputerVisionProjects\\FRs\\ModelEvaluation\\ImageData\\TestImages\\Tarapada_Test\\IMG_20190708_122350.jpg"

    img1= cv2.imread(img1_path, cv2.COLOR_BGR2RGB)
    img1= cv2.resize(img1, (825, 700))
    # img1= imutils.resize(img1, width=700)
    
    img2= cv2.imread(img2_path, cv2.COLOR_BGR2RGB)
    img2= cv2.resize(img2, (825, 700))
    # img2= imutils.resize(img2, width=700)

    img_batch= [img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2, img1,img2]
    # img_batch= [img1, img2]

    # img_batch= cv2.dnn.blobFromImages(img_batch, size= (700,825))
    # print(img_batch.shape)

    ########### batch_embedding
    batch_embeddings, batch_landmark= get_batch_embedding(img_batch)
    print(len(batch_embeddings), len(batch_landmark))
    # for facess_emb, faces_landmarks, img in zip(batch_embeddings, batch_landmark, img_batch):
    #     for face_embedding, face_landmark in zip(facess_emb, faces_landmarks):
    #         bb= face_landmark.rect
    #         sau.markFaces(img, [bb.top(), bb.right(), bb.bottom(), bb.left()], "name", 90, 80)
    #     cv2.imshow("", img)
    #     cv2.waitKey(0)

    ######## batch crop
    # start_time= time.time()
    # print(len(get_batch_crop_faces_from_diff_input_size(img_batch)), time.time()-start_time)

    # batch_aling, batch_landmar, no_bace_list= get_batch_crop_faces_from_diff_input_size(img_batch, slient_drop_multiface=False)
    # for img, bbs in zip(img_batch, batch_landmar):
    #     for bb in bbs:
    #         # print(bb.num_parts)
    #         sau.markFaces(img, [bb.rect.top(), bb.rect.right(), bb.rect.bottom(), bb.rect.left()], "name", 90, 80)
    #     cv2.imshow("", img)
    #     cv2.waitKey(0)

    ################ batch image load
    # batch_img_test, _= load_batch_images_from_list([img1_path, img2_path])
    # cv2.imshow("",batch_img_test[0])
    # cv2.waitKey()