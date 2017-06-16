# Advanced_Lane_Finding

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in ".//Advanced_Lane_Finding.ipynb"  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image:
![calibration1](https://user-images.githubusercontent.com/26694585/27219222-90a777da-529e-11e7-852c-26710e72520b.jpg)

Undistorted Image:
![undistortedchess](https://user-images.githubusercontent.com/26694585/27221591-b7559774-52a6-11e7-8371-8f36ca05cca3.jpg)



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
![undistortedroad](https://user-images.githubusercontent.com/26694585/27223560-928011ce-52ae-11e7-9876-676756f5bea9.jpg)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. (thresholding steps in code cell #4 of Advanced_Lane_Finding.ipynb).  The code given below can give you an idea of the color spaces and threshold I used to produce the binary output:

`def` pipeline(img):
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    gradx = sobel_x_thresh(undistorted, sx_thresh = (20,255), kernel_size = 3)
    grady = sobel_y_thresh(undistorted, sy_thresh = (60,255), kernel_size = 3)
    mag_binary = mag_thresh(undistorted, mag_thresh=(40, 255), sobel_kernel=3)
    dir_binary = dir_threshold(undistorted, dir_thresh=(0.7, 1.2), sobel_kernel=3)
    comb_x_mag = np.zeros_like(dir_binary)
    comb_x_mag[((gradx == 1)&(grady==1)) | ((mag_binary == 1)&(dir_binary==1))] = 1
    S_img = S_thresh(undistorted, (90, 255))
    comb_S = np.zeros_like(S_img)
    comb_S[(S_img==1)|(comb_x_mag==1)]=1
    comb_S = ROI(comb_S, [vertices])
    final = cv2.warpPerspective(comb_S, M, (1280,720), flags=cv2.INTER_LINEAR)
    return(final)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is includes in a function called `pipeline()`(shown above) which appears in the 4th code cell of the IPython notebook).  The `pipeline()` function takes as inputs an image (`img`), and outputs the final binary image.  I chose to the hardcode the source and destination points to be used for the `cv2.warpPerspective()`

```
src = np.float32(
        [[585,455],
         [705,455],
         [1050,680],
         [260,680]])

dest = np.float32(
        [[400,0],
         [880,0],
         [880,710],
         [400,710]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 455      | 400,0        | 
| 705,455       | 880,00      |
| 1050,680      | 880,0      |
| 260,680       | 400,710       |

The image below shows the source and destination points on an image with straight lane lines:
The red and blue points are source and destination points repectively:

![src2dest](https://user-images.githubusercontent.com/26694585/27222209-0c80c9b0-52a9-11e7-9f87-9d5df9f20d73.jpg)


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped](https://user-images.githubusercontent.com/26694585/27222224-18c33b86-52a9-11e7-8f48-867903cb62c7.jpg)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Initial Lane identification takes place in the 7th code cell of the jupyter notebook.
Method:
I took a histogram of the bottom half of the image to find the starting points of the lanes. Like this:

![histogram](https://user-images.githubusercontent.com/26694585/27223946-50574f9a-52b0-11e7-9df6-722d0ce3a465.jpg)

Starting from these points I formed a rectangle of 100x50 pixels, and i stored all the points inside this rectangle in a seperate variable, if the number of pixels inside the rectangle was more than 50 then the a new rectangle would be formed at the mean point of all the pixels inside the previous rectangle. This would continue until all the pixel values for the lane lines arent collected. Then these identified co-ordinates (marked in red and blue in the below image) would be passed to the np.polyfit() function to obtain the polynomial coefficients of the lane lines.

![polynomial](https://user-images.githubusercontent.com/26694585/27222237-263f4246-52a9-11e7-81fe-0c0b188d4f1e.jpg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To calcualate these values of the road I wrote a funtion called `get_lane_values()` in the 23rd code cell. This function takes left_fitx, and right_fitx as inputs and returns mean distance between the lanes, centre of vehicle offset, left lane radius and right lane radius. This function is used in displaying the road values on the final output as well as to check the sanity of the new found lane lines.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:


![lanelines](https://user-images.githubusercontent.com/26694585/27222278-45473568-52a9-11e7-9386-7a3230e190b9.jpg)


### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my project video result [https://youtu.be/C-QShn4fnCQ](url)
Here's a link to my challenge video result [https://youtu.be/6CpQ50lcEBA](url)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I had taken a region of interest first and then used different color channels to get a binary image, this works in the S channel but not on sobel operators. In the Sobel operation we divide the image pixels by np.max(abs_sobel), and due to the black part of the image generated after using ROI, this value would be too high and would hence identify incorrect lines on the image.


My pipeline fails in rains and at night when there is a lot of artificial light reflecting of the road. And it also fails on shadows created by objects not on the road.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
