# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.



**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Pipeline Description. 

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then the grayscale image is passed to Gaussian Blurring algorithm where image smoothening is performed. The output from this module is sent to Canny edge detection where a 3x3 kernel is used to find the edges of the lane lines. The output of the edges is passed to region masking algorithm where a RoI (Region of Interest) is choosen and irrelevant region is discarded. It is then passed to probabilistic hough line transform module where lines are detected and lane lines are extrapolated.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by considering the slope of the left and right lane lines and taking the average of the top X and bottom Y coordinates, while fixing the position of Y coordinates. The left lane has a negative slope and the right lane has a positive slope, in the image domain.

The output of the pipeline is stored in "test_videos_output" folder for white and yellow color lane lines.


### 2. Potential shortcomings with the current pipeline


One potential shortcoming would be that the Y coordinates of lane lines are fixed. If this pipeline were to be used on curved roads, then it would lead to wrong lane marking. 

Another shortcoming would be that since it is a computer vision pipeline, the detection of lane lines in night driving conditions would possess a huge challenge!

Futher, in different countries, there could be a possibility of coming across zig-zag lanes, different color lanes, etc which would mean constructing a different pipeline based on different lane pattern. 


### 3. Possible improvements to your pipeline

A possible improvement would be to calculate the lane extrapolation by also varying Y coordinates of lane lines.

Another potential improvement could be using deep neural networks for lane line detection which would lead to better accuracy on day/night and straight/curved roads. One such network which I used is LaneNet.
