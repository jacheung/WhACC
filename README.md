![](./pictures/whacc-logo-v1.png) <br />

WhACC is a tool for automated touched image classification. 

Many neuroscience labs (e.g. [Hires Lab](https://www.hireslab.org/)) use tasks that involve whisker active touch against thin movable poles to study diverse questions of sensory and motor coding. Since neurons operate at temporal resolutions of milliseconds, determining precise whisker contact periods is essential. Yet, accurately classifying the precise moment of touch is time-consuming and labor intensive. 

We propose a fast, accurate, and generalizable solution using TensorFlow's pretrained CNN base model, specifically MobileNetV2, and 400,000 semi-automated curated images. Check out the below walkthrough for package install instructions and how to upload your own dataset for prediction.

Current version 1 requires files to be packaged in H5 files and has functions for image extractions. 
Version 2 (ETA Q4 2021) will utilize a framework for transfer learning with your own data and the Viterbi algorithm for post-process prediction smoothing. 

## [Walkthrough: Google CoLab](https://colab.research.google.com/drive/1pgdpc0IWkce07Sto6AolQTGoXKCW_mes?authuser=1&pli=1#scrollTo=UAIbs6IlTTfj)  

![](./pictures/trial_animation.gif) <br />
*Single example trial lasting 4 seconds. Example video (left) along with whisker traces, decomposed components, and spikes recorded from L5 (right). How do we identify the precise millisecond frame when touch occurs?*

## Data: 
Current dataset involves 400,000 semi-automated curated images. The distribution with sample images are listed below.  
![](./pictures/frame_distribution.png)

## Code contributors:
WhACC code and software was originally developed by Jonathan Cheung and Phillip Maire in the laboratory of [Samuel Andrew Hires](https://www.hireslab.org/). 
