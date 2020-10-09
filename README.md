# Whisker Automated Contact Classifier 

## Premise: 
Many labs (e.g. [Hires Lab](https://www.hireslab.org/)) use tasks that involve whisker active touch against thin movable poles to study diverse questions of sensory and motor coding. Since neurons operate at temporal resolutions of milliseconds, determining precise whisker contact periods is essential. Yet, accurately classifying the precise moment of touch is time-consuming and labor intensive. Current classification techniques rely on semi-automated curation, infrared beam crossings, or hard coded distance to object thresholds. All involve substantial tradeoffs in analysis time or accuracy.  We propose the use of convolutional neural networks to fully automate the identification of whisker-object touch periods.

![](./pictures/trial_animation.gif) <br />
*Single example trial lasting 4 seconds. Example video (left) along with whisker traces, decomposed components, and spikes recorded from L5 (right). How do we identify the precise millisecond frame when touch occurs?*

## Data: 
Current dataset involves 400,000 semi-automated curated images. The distribution with sample images are listed below.  
<div align=”center”> ![](./pictures/frame_distribution.png)</div>

## Walk through: 
A tutorial walk-through using Google CoLab can be viewed here on [Google CoLab](https://colab.research.google.com/drive/1pgdpc0IWkce07Sto6AolQTGoXKCW_mes?authuser=2#scrollTo=6NkUgDlChm79)  
- load_model() : This function builds TensorFlow's MobileNetV2 ands adds a classification head. Trained model parameters, via TensorFlow checkpoints, are automatically loaded into the model so classification can be performed on a new dataset. 
- ImageBatchGenerator() : This is a class for creating an image generator. An image generator's purpose is to batch images. Inputs will be directories for image datasets. 
- model.predict(ImageBatchGenerator) : simply put, this will output prediction probabilities of touch (1) or non-touch (0). 

## Code contributors:
WhACC code and software was originally developed by Jonathan Cheung and Phillip Maire in the laboratory of [Samuel Andrew Hires](https://www.hireslab.org/). 
