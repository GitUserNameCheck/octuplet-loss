import numpy as np
import onnxruntime as rt
import mediapipe as mp
import cv2
import os
import time
from skimage.transform import SimilarityTransform


def getEmbeddings(model_path, img_dir, matrix_path):

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

    FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1
    )
    
    FACE_RECOGNIZER = rt.InferenceSession(model_path, providers=rt.get_available_providers())


    embeddings = []  # To store embeddings for all images
    reduced_embeddings = []  # To store reduced embeddings for all images
    aligned_images = []  # To store aligned images as a batch

    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)

        # Skip non-image files (optional, adjust as necessary)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path)

        print(img_path)

        result = FACE_DETECTOR.process(img)

        if result.multi_face_landmarks:
            # Select 5 Landmarks (Eye Centers, Nose Tip, Left Mouth Corner, Right Mouth Corner)
            five_landmarks = np.asarray(result.multi_face_landmarks[0].landmark)[[470, 475, 1, 57, 287]]

            # Extract the x and y coordinates of the landmarks of interest
            landmarks = np.asarray(
                [[landmark.x * img.shape[1], landmark.y * img.shape[0]] for landmark in five_landmarks]
            )

            # Extract the x and y coordinates of all landmarks
            all_x_coords = [landmark.x * img.shape[1] for landmark in result.multi_face_landmarks[0].landmark]
            all_y_coords = [landmark.y * img.shape[0] for landmark in result.multi_face_landmarks[0].landmark]

            # Compute the bounding box of the face
            x_min, x_max = int(min(all_x_coords)), int(max(all_x_coords))
            y_min, y_max = int(min(all_y_coords)), int(max(all_y_coords))
            bbox = [[x_min, y_min], [x_max, y_max]]

        else:
            print("face not detected", img_name)
            continue

        # Align Image with the 5 Landmarks
        tform = SimilarityTransform()
        tform.estimate(landmarks, LANDMARKS_TARGET)
        tmatrix = tform.params[0:2, :]
        img_aligned = cv2.warpAffine(img, tmatrix, (112, 112), borderValue=0.0)

        aligned_images.append(img_aligned)  # Add aligned image to batch

    if len(aligned_images) == 0:
            return None, None  # No faces detected in any image

    aligned_images_batch = np.asarray(aligned_images).astype(np.float32)
    aligned_images_batch = aligned_images_batch.clip(0.0, 255.0).transpose(0, 3, 1, 2)  # Convert to NCHW format

    # Inference face embeddings with onnxruntime
    embeddings_batch = FACE_RECOGNIZER.run(None, {"input_image": aligned_images_batch})[0]

    projection_matrix = np.load(matrix_path)
    for embedding in embeddings_batch:
            reduced_embedding = np.dot(embedding, projection_matrix)
            embeddings.append(embedding)
            reduced_embeddings.append(reduced_embedding)

    return embeddings, reduced_embeddings


if __name__ == "__main__":

    dir = "C:/Users/howto/Downloads/FaceTransformerOctupletLoss"
    model_path = f"{dir}/FaceTransformerOctupletLoss.onnx"
    img_dir = f"{dir}/data/"
    matrix_path = f"{dir}/matrix.npy"


    embeddings, reduced_embeddings = getEmbeddings(model_path, img_dir, matrix_path)
    print(len(embeddings), len(reduced_embeddings))
    np.save(f"{dir}/results/embeddings_array.npy", embeddings)
    np.save(f"{dir}/results/reduced_embeddings_array.npy", reduced_embeddings)

    first_embedding = np.load(f"{dir}/results/embedding_reduced.npy")

    print(np.array_equal(reduced_embeddings[0], first_embedding))

