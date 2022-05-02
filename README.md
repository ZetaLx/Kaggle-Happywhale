# Kaggle-Happywhale
Kaggle Happywhale - Whale and Dolphin Identification Top2% solution (28 / 1588 place) 

Kaggle Link: https://www.kaggle.com/competitions/happy-whale-and-dolphin

The data includes more than 15,000 unique individual images of marine mammals from 30 different species from 28 different research institutions. Marine researchers have manually identified and given individual labels. For each image, predict the individual id (individual_id), some individuals in the test data are not observed in the training data, these individuals should be predicted as new individuals (new_individual). The evaluation index is MAP@5.


## Pipeline:

![Image Error](https://github.com/ZetaLx/Kaggle-Happywhale/blob/main/Figure/model.png)
<div align=center><img src="https://github.com/ZetaLx/Kaggle-Happywhale/blob/main/Figure/model.png" /></div>

1. Image data preprocessing - iconic feature image cropping: First, train the YOLOv5 target detection model based on the labeled data, and cut out the dorsal fin or body part from the training set and test set data.

2. Dorsal fin image feature extraction model: Divide the training set data into two parts: training and verification, train the EfficientNet-B7 (backone) model, and input the feature layers of the last two modules of the backone into DOLG (orthogonal feature fusion layer) fusion, Using Arcface as the loss function effectively enhances the intra-class compactness and inter-class separation.

3. Pseudo-label noise data fusion: extract the test set data embedding features from the trained model, construct pseudo-label data with part of the test set prediction results according to the confidence of the verification results, and retrain the backone model together with the training part of step 2.

4. Clustering and sorting: Use the backone model completed by the final training to extract the embedded features of the training set and the test set respectively, train the KNN model with the embedded features of the training set, and then infer the distance of the embedded features of the test set, and sort to obtain the top5 category as the final result.


## Reference:

DOLG paper: https://arxiv.org/pdf/2108.02927.pdf

Arcface paper: https://arxiv.org/pdf/1801.07698.pdf
