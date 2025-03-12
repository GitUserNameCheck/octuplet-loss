import numpy as np
import onnxruntime as rt
import mediapipe as mp
import cv2
import os
import time
from skimage.transform import SimilarityTransform

dir = "C:/Users/howto/Downloads/FaceTransformerOctupletLoss"

# ---------------------------------------------------------------------------------------------------------------------
# INITIALIZATIONS

# Target landmark coordinates for alignment (used in training)
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

# Initialize Face Detector (For Example Mediapipe)
FACE_DETECTOR = mp.solutions.face_mesh.FaceMesh(
    refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1
)

# Initialize the Face Recognition Model (FaceTransformerOctupletLoss)
FACE_RECOGNIZER = rt.InferenceSession(f"{dir}/FaceTransformerOctupletLoss.onnx", providers=rt.get_available_providers())


# ---------------------------------------------------------------------------------------------------------------------
# FACE CAPTURE

img = cv2.imread(f"{dir}/data/0.jpeg") # read the frame from disk
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------------------------------------------------
# FACE DETECTION

# Process the image with the face detector
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
    print("No faces detected")
    exit()


# ---------------------------------------------------------------------------------------------------------------------
# FACE ALIGNMENT

# Align Image with the 5 Landmarks
tform = SimilarityTransform()
tform.estimate(landmarks, LANDMARKS_TARGET)
tmatrix = tform.params[0:2, :]
img_aligned = cv2.warpAffine(img, tmatrix, (112, 112), borderValue=0.0)

# safe to disk
# cv2.imwrite(f"{dir}/reference/img_aligned.jpg", img_aligned)


# ---------------------------------------------------------------------------------------------------------------------
# FACE RECOGNITION

# Inference face embeddings with onnxruntime
input_image = (np.asarray([img_aligned]).astype(np.float32)).clip(0.0, 255.0).transpose(0, 3, 1, 2)
embedding = FACE_RECOGNIZER.run(None, {"input_image": input_image})[0][0]
np.save(f"{dir}/results/embedding.npy", embedding)

# print(len(embedding))
# print("Embedding:", embedding)

# projection_matrix = np.random.random((512, 256))
# np.save(f"{dir}/matrix.npy", projection_matrix)

projection_matrix = np.load(f"{dir}/matrix.npy")
embedding_reduced = np.dot(embedding, projection_matrix)
np.save(f"{dir}/results/embedding_reduced.npy", embedding_reduced)
print(embedding_reduced)


# If you have embeddings for several facial images - you can then compute the cosine distance between them and distinguish
# between different or same people based on a threshold. For example, if the cosine distance is less than 0.5, then the
# two images are of the same person, otherwise they are of different people. The lower the cosine distance, the more similar
# the two images are. The cosine distance is a value between 0 and 2, where 0 means the two images are identical and 2 means 
# the two images are completely different. 

# # ---------------------------------------------------------------------------------------------------------------------
# # VISUALIZATION

# # Draw Boundingbox on a copy of image
img_draw = img.copy()
cv2.rectangle(img_draw, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), (255, 0, 0), 2)

# Show the detected face on the image
cv2.imshow("img", img_draw)
cv2.waitKey(0)

# Show the aligned image
cv2.imshow("img", img_aligned)
cv2.waitKey(0)
