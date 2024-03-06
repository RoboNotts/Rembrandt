# detector.py

from pathlib import Path

import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

"""

IMPORTANT: UN-COMMENT THE FOLLOWING CODE TO MAKE RELEVANT DIRECTORIES
Creates Directories

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

"""

# NS - Iterates through every directory in training/, saving the label from each directory and then loading in the image
# hog - stands for 'Histogram of Oriented Gradients' - works best with a CPU
# Can also use CNN (Convolution neural network) - works best with a GPU
def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        # NS - Detects the location of faces in an image, returns a list of 4-element tuples, one tuple per detected face
        # NS - The 4 elements per tuple provide the four coordinates around the face, called a bounding box
        face_locations = face_recognition.face_locations(image, model=model)

        # NS - Generates encodings for the detected faces in an image
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # NS - Add names and encodings to separate lists
        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # NS - Creates a dictionary that puts the names and encodings together
    name_encodings = {"names":names, "encodings": encodings}

    # NS - Saves encodings to a Disk
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

encode_known_faces()