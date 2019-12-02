# FaceRecognitionUsingFaceNet

Steps
  1. Install requirements.txt
  2. Run load_data.py -i <image-folder-path>
     It extracts faces of all the image
  3. Run extract_face_embed.py
     It extracts features of the extracted face using FaceNet model [128 Features]
  4. Run build_annoy_index.py
     It creates a index with respective of face embeddings
  5. Run face_match.py
     Takes a sample image as input and finds any match in the system
