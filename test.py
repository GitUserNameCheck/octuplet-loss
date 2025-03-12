from pytorch_octuplet_loss import pairwise_distances
import torch
import numpy as np

dir = "C:/Users/howto/Downloads/FaceTransformerOctupletLoss"

embeddings = np.load(f"{dir}/results/reduced_embeddings_array.npy")

# tensor([-2.2204e-16,  2.4152e-02,  1.5912e-02,  1.8892e-02,  1.6508e-02,
#          1.9504e-02,  2.4936e-02,  2.1797e-02,  2.0139e-02,  1.7720e-02,
#          1.8078e-02,  1.8422e-02], dtype=torch.float64)
# tensor([True, True, True, True, True, True, True, True, True, True, True, True])
#The result ranges from 0 to 2, where 0 means identical embeddings, and higher values indicate dissimilarity.
distance_to_reference = pairwise_distances(torch.tensor(embeddings), "cosine")[0]
print(distance_to_reference)

threshold = 0.5

same_person = distance_to_reference < threshold
print(same_person)

# distance_to_reference = pairwise_distances(torch.tensor(embeddings), "euclidean_squared")[0]
# print(distance_to_reference)
# distance_to_reference = pairwise_distances(torch.tensor(embeddings), "euclidean")[0]
# print(distance_to_reference)

