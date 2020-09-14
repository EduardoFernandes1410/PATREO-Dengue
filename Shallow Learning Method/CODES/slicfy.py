#Imports
import os
import numpy as np
from joblib import dump
from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage.segmentation import slic
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path_train', type=str)
parser.add_argument('--images_path_test', type=str)
args = parser.parse_args()

#SLIC
def slicfy(img):
    sigma = 5
    n_segments = 2000
    segments = slic(img, sigma=sigma, n_segments=n_segments)
    final = color.label2rgb(segments, img, kind='avg')
    return [final, segments]

#MAIN
def main():
    #Create 'SLIC' dir
    pathS0 = "./IMAGE DATA"
    pathS1 = "./IMAGE DATA/train/"
    pathS2 = "./IMAGE DATA/test/"
    if not os.path.isdir(pathS0):
        os.mkdir(pathS0)
    if not os.path.isdir(pathS1):
        os.mkdir(pathS1)
    if not os.path.isdir(pathS2):
        os.mkdir(pathS2)

    #Iterate over images
    paths = [args.images_path_train, args.images_path_test]
    aux = ["train", "test"]
    for i, y in enumerate(paths):
        path = y
        directory = os.fsencode(path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            print(filename)
            original = io.imread(os.path.join(path, filename))

            slic = slicfy(original)

            # Save segments
            path_save = "./IMAGE DATA/{0}/SLIC/".format(aux[i])
            if not os.path.isdir(path_save):
                os.mkdir(path_save)

            name = filename.split('.')[0]
            dump(slic[1], path_save + "{0}.joblib".format(name))

if __name__ == "__main__":
    main()
