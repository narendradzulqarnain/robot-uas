
Face Recognition - v8 2025-06-12 2:27am
==============================

This dataset was exported via roboflow.com on June 11, 2025 at 7:27 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 351 images.
Face are annotated in folder format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 160x160 (Fit (black edges))
* Grayscale (CRT phosphor)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random rotation of between -15 and +15 degrees
* Random shear of between -4° to +4° horizontally and -4° to +4° vertically
* Random brigthness adjustment of between -30 and +30 percent
* Random exposure adjustment of between -10 and +10 percent


