import os
import sys
import inspect
import matplotlib.pyplot as plt
import PIL
from autocrop import Cropper
from glob import glob

dir='db/test'
dir_path='db/test/'

faces = [f for f in os.listdir(dir)if not f.startswith('c')]
def plot_test_images(faces, cropper):
    """Given a list on filepaths, crops and plots them."""
    for face in faces:
        print(face)
        try:
            img_array = cropper.crop(dir_path+face)
        except:
            # If we don't detect a face, move on to the next one
            pass
        if img_array is not None:
            print('processed: '+face)
            image = PIL.Image.fromarray(img_array)
            image.save(dir+'/c'+face)

plot_test_images(faces, Cropper(face_percent=80))
