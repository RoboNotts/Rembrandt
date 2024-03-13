# Rembrandt
Repo for Facial Recognition

## Packages to install

### Python

1. dlib
2. face-recognition
3. numpy
4. Pillow

To download dlib and face-recognition, your local device needs Visual Studio C++ version to also be installed

### Repository

The code will ask you to install face_recognition_models from the following repository:https://github.com/ageitgey/face_recognition_models

To install using pip: 

    pip install git+https://github.com/ageitgey/face_recognition_models

When running the code, it might give you an error even after you install the git repository. To fix this, run:

    pip install setuptools

The repository should work afterwards.

## How to use

Un-comment out the initial block of code to create the sub-folders needed - these will be training/output/validation

Once these are created, add any images wanted into the training folder - this folder can have sub-folders if needed, and the code should read them accurately.

You then want to run the encode_known_faces() function - this only needs to be run once at the start to train the models. The encodings for the faces will be stored afterwards. If more images are added, the function will have to be rerun.





