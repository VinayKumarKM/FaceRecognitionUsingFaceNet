import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import pandas as pd

# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    face_pixels = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(face_pixels)
    return yhat[0]

if __name__ == "__main__":
    # Load dataset and Pandas
    data = load('./face_embedding/facenet-image-dataset.npz')
    trainX, trainY = data['arr_0'], data['arr_1']

    # Load CSV
    main_db_df = pd.read_csv('./face_embedding_csv/facesdb.csv')
    face_pixels = main_db_df['faces']

    # Empty face embed list to store in DataFrame
    face_embed_features = list()
    # load the facenet model
    model = load_model('./model/facenet_keras.h5')
    print('Loaded Model')

    for each_face in trainX:
        # array_pixel = asarray(each_face)
        embedding = get_embedding(model, each_face)
        face_embed_features.append(embedding)

    # Add new column to df and export
    main_db_df['face_embedding'] = [face_embed_features]
    main_db_df.to_csv('facesdb.csv', index=False)

    # Export New face embeddings
    # save arrays to one file in compressed format
    savez_compressed('./face_embedding/faces-embeddings.npz', trainX, trainY, face_embed_features)