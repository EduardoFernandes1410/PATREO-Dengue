#!/bin/bash
images_path_train=$1
images_path_test=$2
annotation_path_train=$3
annotation_path_test=$4

echo "STARTING slicfy.py"
echo $(date)
python3 ./CODES/slicfy.py --images_path_train $images_path_train --images_path_test $images_path_test
echo "FINISHED slicfy.py"
echo $(date)

echo -e "\nSTARTING svm.py"
echo $(date)
python3 ./CODES/svm.py --images_path $images_path_train --annotation_path $annotation_path_train
echo "FINISHED svm.py"
echo $(date)

echo -e "\nSTARTING hist_classifier.py"
echo $(date)
python3 ./CODES/hist_classifier.py
echo "FINISHED hist_classifier.py"
echo $(date)

echo -e "\nSTARTING svm_test.py"
echo $(date)
python3 ./CODES/svm_test.py --images_path $images_path_test --annotation_path $annotation_path_test
echo "FINISHED svm_test.py"
echo $(date)

echo -e "\nSTARTING segmentation_confidence.py"
echo $(date)
python3 ./CODES/segmentation_confidence.py --images_path $images_path_test
echo "FINISHED segmentation_confidence.py"
echo $(date)
