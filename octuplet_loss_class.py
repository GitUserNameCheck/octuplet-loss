import logging
from typing import List, Tuple

from pathlib import Path
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as rt
from skimage.transform import SimilarityTransform

model_path = Path(__file__).resolve().parents[0] / "FaceTransformerOctupletLoss.onnx"

MAX_NUM_OF_FACES_TO_FIND = 5

LANDMARKS_TARGET = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

class FaceRecognizerOctupletLoss():
    def __init__(self, on_gpu=False) -> None:
        self.logger = logging.getLogger()
        self.face_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=MAX_NUM_OF_FACES_TO_FIND
        )
        self.face_recognizer = rt.InferenceSession(model_path, providers=rt.get_available_providers())
        self.tform = SimilarityTransform()


    def __calculate_embedding_from_face(self, aligned_images: List[np.ndarray]) -> List[np.ndarray]:
            if(len(aligned_images) > 0):
                aligned_images_batch = np.asarray(aligned_images).astype(np.float32)
                aligned_images_batch = aligned_images_batch.clip(0.0, 255.0).transpose(0, 3, 1, 2)
                embeddings_batch = self.face_recognizer.run(None, {"input_image": aligned_images_batch})[0]
                return embeddings_batch
            else:
                return []

    def __align_faces(self, landmarks_arr: List[np.ndarray], _img_rgb: np.ndarray) -> List[np.ndarray]:
        aligned_images = []
        for landmarks_from_one_image, image in zip(landmarks_arr, _img_rgb):
            for landmarks in landmarks_from_one_image:
                self.tform.estimate(landmarks, LANDMARKS_TARGET)
                tmatrix = self.tform.params[0:2, :]
                img_aligned = cv2.warpAffine(image, tmatrix, (112, 112), borderValue=0.0)
                aligned_images.append(img_aligned)
                cv2.imshow("img", img_aligned)
                cv2.waitKey(0)
        return aligned_images
        
    def __find_faces(self, _img_rgb: np.ndarray) -> Tuple[List[List], List[np.ndarray]]:
        face_bbs = []
        landmarks_arr = []
        for img in _img_rgb:
            result = self.face_detector.process(img)
            if result.multi_face_landmarks:
                landmarks_from_one_image_arr = []
                for landmark in result.multi_face_landmarks:
                    # Select 5 Landmarks (Eye Centers, Nose Tip, Left Mouth Corner, Right Mouth Corner)
                    five_landmarks = np.asarray(landmark.landmark)[[470, 475, 1, 57, 287]]

                    # Extract the x and y coordinates of the landmarks of interest
                    landmarks = np.asarray(
                        [[landmark.x * img.shape[1], landmark.y * img.shape[0]] for landmark in five_landmarks]
                    )

                    # Extract the x and y coordinates of all landmarks
                    all_x_coords = [landmark.x * img.shape[1] for landmark in landmark.landmark]
                    all_y_coords = [landmark.y * img.shape[0] for landmark in landmark.landmark]

                    # Compute the bounding box of the face
                    x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
                    y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
                    bbox = [[x_min, y_min], [x_max, y_max]]

                    img_draw = img.copy()
                    cv2.rectangle(img_draw, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (255, 0, 0), 2)

                    # Show the detected face on the image
                    cv2.imshow("img", img_draw)
                    cv2.waitKey(0)

                    print(landmarks)

                    face_bbs.append([x_min, y_min, x_max, y_max])
                    landmarks_from_one_image_arr.append(landmarks)
                landmarks_arr.append(landmarks_from_one_image_arr)  
            else:
                print("no result")         
        return face_bbs, landmarks_arr

    def __calculate_embeddings(self, _img_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        face_bbs, landmarks_arr = self.__find_faces(_img_rgb)
        aligned_images = self.__align_faces(landmarks_arr, _img_rgb)
        return self.__calculate_embedding_from_face(aligned_images), face_bbs

    def get_image_embeddings(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        return self.__calculate_embeddings(image)



if __name__ == "__main__":

    dir = "C:/Users/howto/Downloads/NN/FaceTransformerOctupletLoss"

    img1 = cv2.imread(f"{dir}/data/1427491329.jpg")
    img2 = cv2.imread(f"{dir}/data/multi_face.jpg")

    images = [img1, img2]

    # print("img1", img1)

    # print("img2", img2)

    recognizer = FaceRecognizerOctupletLoss()

    embeddings, face_bb = recognizer.get_image_embeddings(images)

    print(embeddings)
    print(face_bb)
