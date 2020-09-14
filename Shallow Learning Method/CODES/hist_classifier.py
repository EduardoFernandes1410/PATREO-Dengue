import numpy as np
import os
import skimage
from joblib import dump, load
from skimage import io, img_as_uint
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler

# #### Read Histogram and GLCM data
print("vou loadar")
###### TODO ######
c_hist = load("./DATA/c_hist.joblib")
c_gt = load("./DATA/c_gt.joblib")

# #### Scale features
print("vou escalar")
scaler = StandardScaler()
scaled_full = scaler.fit_transform(c_hist)
dump(scaler, "./DATA/scaler.joblib")

# #### Create classifier
print("vou treinar")
clf_full = LinearSVC(max_iter=10000, dual=False)
clf_full.fit(scaled_full, c_gt)

dump(clf_full, "./DATA/clf_full.joblib")

