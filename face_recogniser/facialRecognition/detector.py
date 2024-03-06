# detector.py

from pathlib import Path

import face_recognition

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

"""
Creates Directories

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

"""

# Iterates through every directory in training/, saving the label from each directory and then loading in the image
# hog - stands for 'Histogram of Oriented Gradients' - works best with a CPU
# Can also use CNN (Convolution neural network) - works best with a GPU

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        # Uses encoding - an array of numbers describing the features of the face
        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            name.append(name)
            encodings.append(encoding)