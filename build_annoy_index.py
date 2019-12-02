from annoy import AnnoyIndex
from numpy import load
import pandas as pd

def build_annoy_index(encoding_dim, num_trees, annoy_index_file, encodings):
    ann = AnnoyIndex(encoding_dim, 'angular')
    for index, encoding in enumerate(encodings, 1):
        ann.add_item(index, encoding)
    # builds a forest of num_trees, higher the number of trees -> higher the precision
    ann.build(num_trees)
    # save the index to a file
    ann.save(annoy_index_file)
    print("Created Annoy Index Successfully")


if __name__ == "__main__":
    data = load('./face_embedding/faces-embeddings.npz')
    trainX, trainy, face_embed = data['arr_0'], data['arr_1'], data['arr_2']
    encoding_vector_length = face_embed.shape[1]
    annoy_file_name = './annoy_index/face-image.annoy.index'
    num_trees = 10
    build_annoy_index(encoding_vector_length, num_trees, annoy_file_name, face_embed)