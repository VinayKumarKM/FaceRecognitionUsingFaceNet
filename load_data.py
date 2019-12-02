# function for face detection with mtcnn
from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os.path


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    try:
        # load image from file
        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    except:
        return None


# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    labels = list()
    # enumerate files
    count = 0
    for filename in listdir(directory):
        # path
        path = directory + '/'+ filename
        # get face
        face = extract_face(path)
        # store
        if face is not None:
            faces.append(face)
            labels.append(filename)
            count += 1
            print(count)
    print('length of face', len(faces))
    print('length of labels', len(labels))
    return faces, labels

if __name__ == "__main__":
    parser = ArgumentParser(description="File path for image")
    parser.add_argument("-i", dest="filename", required=True,
                        help="input file with two matrices")
    args = parser.parse_args()

    db_faces = list()
    db_labels = list()
    db_ids = list()
    file_path = args.filename
    trainX, trainy = load_faces(file_path)
    import ipdb;ipdb.set_trace()
    db_faces.extend(trainX)
    db_labels.extend(trainy)
    db_ids = [i for i in range(1, len(db_faces) + 1)]
    # save arrays to one file in compressed format
    main_db_df = pd.DataFrame(list(zip(db_ids, db_labels, db_faces)), columns=['id', 'labels', 'faces'])
    # main_db_df['faces'] = db_faces.apply(np.array)
    print(main_db_df.head())
    main_db_df.to_csv('./face_embedding_csv/facesdb.csv', index=False)
    savez_compressed('./face_embedding/facenet-image-dataset.npz', trainX, trainy)