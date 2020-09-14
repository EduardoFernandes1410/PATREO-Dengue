# Imports
import numpy as np
import os
import skimage
import histogram
from joblib import dump, load
from skimage import io
from sklearn.svm import LinearSVC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str)
parser.add_argument('--annotation_path', type=str)
args = parser.parse_args()


# Get histogram from each image
def getHistograms():
    complete_hist = np.array([range(64)])
    complete_gt = np.array([])

    path1 = args.images_path
    path2 = "./IMAGE DATA/test/SLIC/"
    path3 = args.annotation_path
    path4 = "./IMAGE DATA/test/HISTOGRAM/"
    if not os.path.isdir(path4):
        os.mkdir(path4)
    directory = os.fsencode(path3)

    #Read images
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        print(filename)
        original_img = io.imread(os.path.join(path1, filename))
        annotated_img = io.imread(os.path.join(path3, filename))
        segments = load(path2 + filename.split('.')[0] + ".joblib")

        # Get image hists
        hist = histogram.main(original_img, segments)

        #Get binary value of each segment
        ground_truth = np.zeros(np.unique(segments).shape[0])
        for segVal in np.unique(segments):
            mask = segments == segVal
            ground_truth[segVal] = (annotated_img[mask].mean() > 128)

        # Save data
        dump(hist, path4 + filename.split('.')[0] + "-HIST.joblib")
        dump(ground_truth, path4 + filename.split('.')[0] + "-GT.joblib")

def main():
    # Get data
    getHistograms()

if __name__ == "__main__":
    main()
