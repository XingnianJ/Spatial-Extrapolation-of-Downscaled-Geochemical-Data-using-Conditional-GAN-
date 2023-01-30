# Spatial extrapolation of downscaled geochemical data based on conditional GAN

We provide interpolated and normalized grid images and related instructions to help you run the following code.The files we need to use include the coarse data and fine data in the training area, as well as the coarse data in the verification area. The former is in the path of '/training area/images exported from ArcGIS', and the latter is in'/verification area/images exported from ArcGIS'.

segment image.py:
A sliding window with the size of 16 * 16 is used to segment the tif image exported from ArcGIS into many sub images according to the set step size. Note that Images exported from Arcgis must maintain the same format, such as 53 * 51 * 1 (height * width * number of channels). In addition, when naming the file, ensure that the fine data comes first and the coarse data comes last.

match image.py:
Match the segmented images to the same folder according to the coordinates named by the segement image.py. After matching, each file is a training/verification sample.

change to tfrecords file.py:
Convert the matched samples into files that can be read by the tensorflow framework.

train.py:
Train a conditional GAN model.

simulation.py:
Call the model saved in training process to generate simulation results.

