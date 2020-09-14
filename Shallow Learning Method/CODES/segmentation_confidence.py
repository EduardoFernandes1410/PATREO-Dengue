# Imports
import os
import cv2
import skimage
import numpy as np
from scipy import ndimage
from joblib import dump, load
from skimage import io, img_as_uint
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str)
args = parser.parse_args()

# Get image mask
def finalImageMask(res, segs):
    mask = []
    for i, x in enumerate(res):
         if(x == 1.0):
            mask += [i]
    mask = np.array(mask)
    mascara = np.isin(segs, mask)
    return mascara

# Get confidence mask
def getConfidenceMask(segs, confidence):
    mascara = np.zeros(segs.shape)
    for i, x in enumerate(confidence):
        if(x > 0):
            mask = segs == i
            mascara[mask] = x

    return mascara

# Get Pool Segments
def getPoolCrops(img_binary):
    # Label objects
    img_labeled, num_features = ndimage.label(img_binary)
    # Find the location of all objects
    objs = ndimage.find_objects(img_labeled)
    return objs

# Make Bounding Boxes
def makeBoundingBoxes(img, img_binary, img_confidence):
    img_bb = np.copy(img)
    img_objs = getPoolCrops(img_binary)
    bounding_boxes = []

    for obj in img_objs:
        start = (obj[1].start, obj[0].start)
        end = (obj[1].stop, obj[0].stop)

        counter = 0
        confidence_value = 0

        for i in range(start[1], end[1]):
            for j in range(start[0], end[0]):
                if img_confidence[i,j] > 0:
                    confidence_value += img_confidence[i,j]
                    counter += 1

        confidence_value /= counter

        img_bb = cv2.rectangle(img_bb, start, end, (0,255,0), 2)
        bounding_boxes += [[start, end, confidence_value]]

    return (img_bb, bounding_boxes)

# Make Bounding Boxes TXT
def makeBoundingBoxesTXT(bounding_boxes, path):
    file = open(path, 'w')
    for row in bounding_boxes:
        file.write("target " + str(row[2]) + " " + str(row[0][0]) + " " + str(row[0][1]) + " " + str(row[1][0]) + " " + str(row[1][1]) + "\n")
    file.close()

# Get histogram from each image
def segment():
    # Get classifier
    clf = load("./DATA/clf_full.joblib")

    path1 = "./IMAGE DATA/test/HISTOGRAM/"
    path2 = "./IMAGE DATA/test/BinarySVC/"
    path3 = "./IMAGE DATA/test/SLIC/"
    path4 = args.images_path
    path5 = "./IMAGE DATA/test/BoundingBoxSVC/"
    path6 = "./IMAGE DATA/test/BoundingBoxTXT/"

    if not os.path.isdir(path2):
        os.mkdir(path2)
    if not os.path.isdir(path5):
        os.mkdir(path5)
    if not os.path.isdir(path6):
        os.mkdir(path6)

    directory = os.fsencode(path4)

    #Read histograms
    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        img = io.imread(os.path.join(path4, filename))
        img_hist = load(path1 + filename.split('.')[0] + "-HIST.joblib")
        img_segs = load(path3 + filename.split('.')[0] + ".joblib")

        # Scale features
        scaler = load("./DATA/scaler.joblib")
        scaled_features = scaler.transform(img_hist)

        # Classify
        res = clf.predict(scaled_features)
        confidence = clf.decision_function(scaled_features)

        # Generate binary image
        mask = finalImageMask(res, img_segs)
        img_binary = np.full(img_segs.shape[:2], False)
        img_binary[mask] = True
        io.imsave(path2 + filename, img_as_uint(img_binary))

        # Generate confidence mask
        img_confidence = getConfidenceMask(img_segs, confidence)

        # Generate bounding boxes
        img_bb, bounding_boxes = makeBoundingBoxes(img, img_binary, img_confidence)
        makeBoundingBoxesTXT(bounding_boxes, path6 + filename.split('.')[0] + ".txt")
        io.imsave(path5 + filename, img_bb)


def main():
    segment()

if __name__ == "__main__":
    main()
