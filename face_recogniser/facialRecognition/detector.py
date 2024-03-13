# detector.py
from pathlib import Path
from collections import Counter
from PIL import Image, ImageDraw

import face_recognition
import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOUR = "blue"
TEXT_COLOUR = "white"


"""

IMPORTANT: UN-COMMENT THE FOLLOWING CODE TO MAKE RELEVANT DIRECTORIES
Creates Directories

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)o
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

#NS - Recognises faces in images that don't have a label
def recognise_faces(image_location: str, model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH,) -> None:
    with encodings_location.open(mode="rb") as f:
        #NS - Create test image
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    #NS - Detect faces in the input image and get their encodings
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    #NS - Creates pillow image object from loaded input image
    pillow_image = Image.fromarray(input_image)

    #NS - Helps draw bounding box around detected faces
    draw = ImageDraw.Draw(pillow_image)

    #NS - iterate through locations and encodings in parallel
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognise_face(unknown_encoding, loaded_encodings)
        #NS - Checks if the name has a match - if nothing is returned, change the name to Unknown
        if not name:
            name = "Unknown"

        #NS - shows name and bounding box size
        #print(name, bounding_box)

        _display_face(draw, bounding_box, name)

    #NS - Housekeeping for Pillow
    del draw
    pillow_image.show()

#NS - Takes the unknown encoding and loaded encodings, and makes a comparison between the unkown encoding
# and the loaded encodings
#NS - Will return the most likely match
def _recognise_face(unknown_encoding, loaded_encodings):
    #NS - compare_faces returns a list of true and false values for each loaded encoding
    #NS - The indicies of this list are equal to those of the loaded encodings
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)

    #NS - Tracks how many votes each potential match has
    #NS - The unknown face is compared to every known face that you have encodings for
    #   - Each match counts as a vote for the person with the known face
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

def _display_face(draw, bounding_box, name):
    #NS - Unpack bounding_box into four parts, and draw rectangle around detected face
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top),(right,bottom)),outline=BOUNDING_BOX_COLOUR)

    #NS - Determines bounding-box for caption
    text_left, text_top, text_right, text_bottom = draw.textbbox((left,bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill="blue", outline="blue")
    draw.text((text_left, text_top), name, fill="white")
'''
IMPORTANT: ONLY RUN THE FOLLOWING LINE ONE TIME (unless more pictures are added)
'''
#encode_known_faces()

recognise_faces("unknown.jpg")