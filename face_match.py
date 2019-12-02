from os import listdir
from os.path import isdir
from PIL import Image
from numpy import savez_compressed
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from annoy import AnnoyIndex
from numpy import load
from keras.models import load_model
import pandas as pd
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import time

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
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


# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')

	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std

	# transform face into one sample
	samples = expand_dims(face_pixels, axis=0)

	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

if __name__ == "__main__":
    parser = ArgumentParser(description="File path for image")
    parser.add_argument("-i", dest="filename", required=True,
                        help="input file with two matrices")
    args = parser.parse_args()
    import ipdb;ipdb.set_trace()
    file_path = args.filename
    sample_image_array = extract_face(file_path)

    # load the facenet model
    model = load_model('./model/facenet_keras.h5')
    print('Loaded Model')

    # Get embed of sample image
    sample_image_embed = get_embedding(model, sample_image_array)
    encoding_vector_length = sample_image_embed.shape[0]

    # Load the annoy index
    annoy_file_name = './annoy_index/face-image.annoy.index'
    saved_ann = AnnoyIndex(encoding_vector_length)
    saved_ann.load(annoy_file_name)

    # get_nns_by_vector returns the indices of the most similar images
    nn_indices = saved_ann.get_nns_by_vector(sample_image_embed, 5)

    # DataFrame for getting the index
    main_db_df = pd.read_csv('./face_embedding_csv/facesdb.csv')
    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(2, 2),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    import ipdb;ipdb.set_trace()
    for i, index in enumerate(nn_indices, 1):
        print(index)
        print("Target Image", main_db_df.iloc[index - 1])