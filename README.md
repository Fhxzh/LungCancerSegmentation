# Lung Cancer Segmentation
This convolutional neural network is concerned with segmenting nodule candidates from ct scans using the data provided by the LUNA16 competition. It constitutes the first part of a bigger project that also involes a network for false positives reduction.

Unet_Segment_wAugm.py is the final script used for training and predicting

Training loss ended up at around 74% and validation loss at around 67%. In the final test run the network was able to achieve an average dice score of 61%.

Predicted masks can be inspected via other/inspect_masks.ipynb

The folder 'preprocessing' contains all the scripts used for preparing the data for the network.
The folder 'other' contains code snippets about k-fold cross validation and the smaller model. The weights file was not pushed to the repo as its size was too big.
