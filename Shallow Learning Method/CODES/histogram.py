#Imports
from skimage.exposure import histogram
import numpy as np

#Discretize pixels in 3D space
def discretize(img):
    discrete_img = np.zeros(img.shape[:2])

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            discrete_img[x][y] = int((img[x][y][0] / 64)) + 4*int((img[x][y][1] / 64)) + 16*int((img[x][y][2] / 64))

    discrete_img = discrete_img.astype(int)
    return discrete_img

#Separate segments
def getSegments(segments, discrete_img):
    segs_img = [[]]

    for segVal in np.unique(segments):
        mask = segments == segVal
        segs_img += [discrete_img[mask]]

    return segs_img

#Make histogram for each segment
def make_hists(segs):
    hists = np.array([range(64)])

    for seg in segs[1:]:
        y,x = histogram(seg, nbins=64)
        y = np.insert(y, 0, np.zeros(x[0]))
        y = np.append(y, np.zeros(64 - len(y)))
        hists = np.append(hists, [y], axis=0)

    return hists[1:]

def main(img, segments):
    discrete_img = discretize(img)
    segs_img = getSegments(segments, discrete_img)
    img_hists = make_hists(segs_img).astype(int)
    return img_hists

if __name__ == "__main__":
    main()
