## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./WriteupResources/CarNotCar.png
[image2]: ./WriteupResources/HOGFeatures.png
[image3]:./WriteupResources/AllRectangles.png
[image5]: ./test_image.jpg
[image6]: ./WriteupResources/AllDetectedVeh.png
[image7]: ./WriteupResources/Heatmap.png
[image8]: ./WriteupResources/DetectedVeh.png

[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=14`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]

I used the get_hog_features function in lines 22 - 41 of the lesson_functions.py file. An example of this is shown in cell 3 of the jupyter notebook VehicleDetection.ipyn


#### 2. Explain how you settled on your final choice of HOG parameters.

I started by training SVM's with just color features and HOG features seperatley. I tried different combinations of parameters and choose parameters based on the best accuracy and the lowest feature count.
Below is the results for just the LUV color space of testing different parameters on the color features:

| Color | Spatial | hist bin | feat size | train t | Test Acc | FP  | FN  |
| ---   | ---     | ---      | ---       | ---     | ---      | --- | --- |
|  LUV  |    8    |    32    |    288    |   4.51  |  0.9547  | 2.14| 2.39|
|  LUV  |    8    |    48    |    336    |   4.61  |  0.9552  | 2.00| 2.48|
|  LUV  |    8    |    64    |    384    |   4.71  |  0.9642  | 1.41| 2.17|
|  LUV  |    8    |    72    |    408    |   4.70  |  0.9583  | 1.97| 2.20|
|  LUV  |    12   |    32    |    528    |   6.95  |  0.9645  | 1.24| 2.31|
|  LUV  |    12   |    48    |    576    |   6.64  |  0.9623  | 1.58| 2.20|
|  LUV  |    12   |    64    |    624    |   6.35  |  0.9566  | 1.60| 2.73|
|  LUV  |    12   |    72    |    648    |   6.50  |  0.9600  | 1.32| 2.67|
|  LUV  |    16   |    32    |    864    |   8.20  |  0.9462  | 2.08| 3.29|
|  LUV  |    16   |    48    |    912    |   7.93  |  0.9521  | 1.91| 2.87|
|  LUV  |    16   |    64    |    960    |   7.59  |  0.9569  | 1.66| 2.65|
|  LUV  |    16   |    72    |    984    |   6.90  |  0.9628  | 1.21| 2.51|
|  LUV  |    20   |    32    |    1296   |   8.31  |  0.9403  | 2.45| 3.52|
|  LUV  |    20   |    48    |    1344   |   8.21  |  0.9400  | 1.91| 4.08|
|  LUV  |    20   |    64    |    1392   |   7.38  |  0.9437  | 1.83| 3.80|
|  LUV  |    20   |    72    |    1416   |   7.16  |  0.9420  | 1.77| 4.03|
|  LUV  |    24   |    32    |    1824   |  11.33  |  0.9358  | 1.58| 4.84|
|  LUV  |    24   |    48    |    1872   |  10.43  |  0.9451  | 1.69| 3.80|
|  LUV  |    24   |    64    |    1920   |  10.05  |  0.9375  | 1.63| 4.62|
|  LUV  |    24   |    72    |    1944   |   9.64  |  0.9451  | 1.44| 4.05|

Below are the results for just the YUV color space of testing different parameters on the HOG features:

| CHog  | orients | pix cell | feat size | train t | Test Acc | FP  | FN  |
| ---   | ---     | ---      | ---       | ---     | ---      | --- | --- |
|  YUV  |    8    |     8    |    1764   |   7.40  |  0.9555  | 2.53| 1.91|
|  YUV  |    8    |     9    |    1764   |   7.15  |  0.9502  | 2.53| 2.45|
|  YUV  |    8    |    10    |    1764   |   7.39  |  0.9488  | 2.67| 2.45|
|  YUV  |    8    |    11    |    1764   |   7.17  |  0.9510  | 2.39| 2.51|
|  YUV  |    8    |    12    |    1764   |   7.41  |  0.9485  | 2.39| 2.76|
|  YUV  |    8    |    13    |    1764   |   7.20  |  0.9454  | 2.82| 2.65|
|  YUV  |    8    |    14    |    1764   |   6.98  |  0.9445  | 2.36| 3.18|
|  YUV  |    8    |    15    |    1764   |   7.39  |  0.9513  | 2.73| 2.14|
|  YUV  |    8    |    16    |    1764   |   7.25  |  0.9457  | 2.51| 2.93|
|  YUV  |    9    |     8    |    1764   |   7.55  |  0.9535  | 2.42| 2.22|
|  YUV  |    9    |     9    |    1764   |   7.27  |  0.9426  | 3.18| 2.56|
|  YUV  |    9    |    10    |    1764   |   7.24  |  0.9513  | 2.28| 2.59|
|  YUV  |    9    |    11    |    1764   |   7.15  |  0.9445  | 2.73| 2.82|
|  YUV  |    9    |    12    |    1764   |   7.23  |  0.9426  | 2.73| 3.01|
|  YUV  |    9    |    13    |    1764   |   7.41  |  0.9507  | 2.28| 2.65|
|  YUV  |    9    |    14    |    1764   |   7.31  |  0.9510  | 2.45| 2.45|
|  YUV  |    9    |    15    |    1764   |   7.02  |  0.9437  | 3.10| 2.53|
|  YUV  |    9    |    16    |    1764   |   7.27  |  0.9457  | 2.79| 2.65|
|  YUV  |    10   |     8    |    1764   |   7.32  |  0.9538  | 2.48| 2.14|
|  YUV  |    10   |     9    |    1764   |   7.25  |  0.9482  | 2.39| 2.79|
|  YUV  |    10   |    10    |    1764   |   7.33  |  0.9485  | 2.22| 2.93|
|  YUV  |    10   |    11    |    1764   |   7.49  |  0.9496  | 2.45| 2.59|
|  YUV  |    10   |    12    |    1764   |   7.15  |  0.9499  | 2.20| 2.82|
|  YUV  |    10   |    13    |    1764   |   7.15  |  0.9462  | 2.73| 2.65|
|  YUV  |    10   |    14    |    1764   |   7.26  |  0.9516  | 2.31| 2.53|
|  YUV  |    10   |    15    |    1764   |   7.52  |  0.9476  | 2.56| 2.67|
|  YUV  |    10   |    16    |    1764   |   7.62  |  0.9462  | 2.34| 3.04|
|  YUV  |    11   |     8    |    1764   |   7.59  |  0.9502  | 2.67| 2.31|
|  YUV  |    11   |     9    |    1764   |   7.10  |  0.9426  | 2.73| 3.01|
|  YUV  |    11   |    10    |    1764   |   7.38  |  0.9457  | 2.42| 3.01|
|  YUV  |    11   |    11    |    1764   |   7.38  |  0.9535  | 2.03| 2.62|
|  YUV  |    11   |    12    |    1764   |   6.90  |  0.9459  | 2.36| 3.04|
|  YUV  |    11   |    13    |    1764   |   7.48  |  0.9451  | 2.48| 3.01|
|  YUV  |    11   |    14    |    1764   |   7.17  |  0.9428  | 2.76| 2.96|
|  YUV  |    11   |    15    |    1764   |   7.24  |  0.9434  | 3.18| 2.48|
|  YUV  |    11   |    16    |    1764   |   7.44  |  0.9471  | 2.70| 2.59|
|  YUV  |    12   |     8    |    1764   |   7.54  |  0.9474  | 2.96| 2.31|
|  YUV  |    12   |     9    |    1764   |   7.32  |  0.9516  | 2.22| 2.62|
|  YUV  |    12   |    10    |    1764   |   7.53  |  0.9471  | 2.36| 2.93|
|  YUV  |    12   |    11    |    1764   |   6.99  |  0.9505  | 2.48| 2.48|
|  YUV  |    12   |    12    |    1764   |   7.10  |  0.9471  | 2.31| 2.98|
|  YUV  |    12   |    13    |    1764   |   7.62  |  0.9502  | 2.65| 2.34|
|  YUV  |    12   |    14    |    1764   |   7.30  |  0.9479  | 2.45| 2.76|
|  YUV  |    12   |    15    |    1764   |   7.21  |  0.9462  | 2.62| 2.76|
|  YUV  |    12   |    16    |    1764   |   7.54  |  0.9493  | 2.56| 2.51|
|  YUV  |    13   |     8    |    1764   |   7.29  |  0.9420  | 3.24| 2.56|
|  YUV  |    13   |     9    |    1764   |   7.19  |  0.9459  | 2.20| 3.21|
|  YUV  |    13   |    10    |    1764   |   7.58  |  0.9524  | 2.48| 2.28|
|  YUV  |    13   |    11    |    1764   |   7.39  |  0.9544  | 2.62| 1.94|
|  YUV  |    13   |    12    |    1764   |   7.35  |  0.9502  | 2.59| 2.39|
|  YUV  |    13   |    13    |    1764   |   6.83  |  0.9471  | 1.97| 3.32|
|  YUV  |    13   |    14    |    1764   |   7.53  |  0.9510  | 2.59| 2.31|
|  YUV  |    13   |    15    |    1764   |   7.24  |  0.9513  | 2.39| 2.48|
|  YUV  |    13   |    16    |    1764   |   7.14  |  0.9479  | 2.14| 3.07|
|  YUV  |    14   |     8    |    1764   |   7.22  |  0.9485  | 2.42| 2.73|
|  YUV  |    14   |     9    |    1764   |   7.29  |  0.9485  | 2.51| 2.65|
|  YUV  |    14   |    10    |    1764   |   7.20  |  0.9535  | 2.20| 2.45|
|  YUV  |    14   |    11    |    1764   |   7.04  |  0.9462  | 2.62| 2.76|
|  YUV  |    14   |    12    |    1764   |   7.41  |  0.9457  | 2.31| 3.12|
|  YUV  |    14   |    13    |    1764   |   7.33  |  0.9465  | 2.59| 2.76|
|  YUV  |    14   |    14    |    1764   |   7.50  |  0.9535  | 1.91| 2.73|
|  YUV  |    14   |    15    |    1764   |   7.47  |  0.9519  | 2.51| 2.31|
|  YUV  |    14   |    16    |    1764   |   7.25  |  0.9507  | 2.31| 2.62|

I also did testing on all the parameters for a SVM using both color and HOG features, which can be found in cell 9 and 10 of the Jupyter Notebook VehicleDetection.ipynb The final results are below

| Color | CHog  | Spatial | hist bin | orients | pix cell | HOG time | feat size | train t | Test Acc | FP  | FN  |
| ---   | ---   | ---     | ---      | ---     | ---      | ---      | ---       | ---     | ---      | --- | --- |
|  LUV  |  YUV  |    20   |    64    |    14   |    16    |   47.95  |    2904   |   4.58  |  0.9935  | 0.17| 0.48|

The full results are in the Training_Regression.ods file
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the provided training images. I used the extract_features function in lines 70-132 of the lesson_functions.py file. Training the SVM was done in cell 12 of the Jupyter notebook VehicleDetection.ipyn

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the find_cars function from the lectures to implement a sliding window search on each image. I used different scaling factors through trial and error, and tried to minimize the regions that each window size was used on the image to reduce the time it took. Below is an image of all the windows used.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on seven scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. To optimize performance I ran many regressions to try to pick the best combination of parameters. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I created a classs to hold the 6 frames of detected vehicles. Then I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the detected vehicles for a test image with the heatmap generated and the final vehicle detection boxes shown

### Original Image:

![alt text][image5]

### Image with all detected vehicles:
![alt text][image6]

### Heatmap:
![alt text][image7]

### Final vehicel detection:
![alt text][image8]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My approach was to use the functions provided in the lecture and adjust the parameters to get a good classifier which I then could use on the video stream. My thought was with a good classifier I wouldn't have to worry too much about handling exceptions in the video. This worked well, but I felt that this was a poor way to learn and I was very dependent on the provided functions rather than thinkig of ways to do this on my own. Ultimatley I would say this project and lectures series encourages a poor learning experience and I would not go through most of the lecture if I were to redo this project.
