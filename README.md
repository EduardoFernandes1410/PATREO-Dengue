# Water Tanks and Swimming Pools Detection in Satellite Images: Exploiting Shallow and Deep-Based Strategies

## Abstract:
  This paper aims to study and to evaluate two distinct approaches for detecting water tanks and swimming pools in satellite images, which can be useful to monitor water-related diseases. The first approach, shallow, consists of using a Support Vector Machine in order to classify into positive and negative a discretized color histogram of a given segment of the original image. The second method employs the Faster R-CNN framework for detecting those objects. We built up swimming pools and water tanks datasets over the city of Belo Horizonte to support our experimental analysis. Our results show that the deep learning method greatly outperforms the shallow strategy, achieving an average precision at 0.5 IoU of over 93\% on the swimming pool detection task, and over 73\% on the water tank one. All the code and datasets are publicly available.
  
## Dataset download: 
http://www.patreo.dcc.ufmg.br/bh-pools-watertanks-datasets/

## Running shallow learning method:
`bash script.sh <path_to_training_images> <path_to_testing_images> <path_to_training_annotation> <path_to_testing_images>`
