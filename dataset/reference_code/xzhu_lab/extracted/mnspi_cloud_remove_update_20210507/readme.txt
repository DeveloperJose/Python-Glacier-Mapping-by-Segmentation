;-------------------------------------------------------------------------------------------------
;                REMOVE THICK CLOUD IN SATELLITE IMAGES
;
;               This code can be used for whole TM scene
;                  Developed and coded by Zhu Xiaolin
;           Department of Geography,the Ohio State University
;                    Email:zhuxiaolin55@gmail.com
;                           updated£º2013-10-23
;     Debug history:
;         2013-10-23 correct:1) the bug of similar pixel searching
;                            2) the bug of using wrong values in cloudy image
;         2017-11-21£ºcorrect a nonstop loop bug. It happens when processing large cloud patches and 
;                     no similar pixel selected for a cloudy pixel
;         2021-05-09£ºcorrect a bug of similar pixel searching for line-shape clouds 
;                            
;     Please cite: Zhu, X., Gao, F., Liu, D. and Chen, J. A modified 
;     neighborhood similar pixel interpolator approach for removing 
;     thick clouds in Landsat images. IEEE Geoscience and Remote 
;     Sensing Letters, 2012, 9(3), 521-525
;     NOTE: the efficiency may be low for large image.If so, please use CLOUD_REMOVE_FAST, 
;     in which some process was simplified, but the accuracy is still satisfactory
;-------------------------------------------------------------------------------------------------

NOTE: CLOUD_REMOVE_FAST.PRO is a faster version of CLOUD_REMOVE.PRO. It simplified some process but the accuracy is still acceptable!

                 CLOUD_REMOVE_FAST.PRO only needs about 1/8 computing time of CLOUD_REMOVE.PRO



1. The program is written in IDL;

2. Before running the program, ENVI should be opened because the program uses some functions of ENVI£»

3. set Parameters 
;------------------------------------------------
 num_class=4  ;set the estimated number of classes in image
 min_pixel=20 ;set the sample size of similar pixels
 extent1=1    ;set the range of cloud neighborhood
 DN_min=0     ;set the range of DN
 DN_max=255
 patch_long=500 ;set the block size when process large images, 500~1000 recommend
 ;-------------------------------------------------

4. Input the images accoring to the name of window: the could_mask image is a could mask in which each cloud is manually delineated and marked by different numbers from 1,2,3.....
Hint: using ENVI, it can be done by ROI for each cloud patches, and then transfer rois to classification map to get the could mask image.  


5. Output the result

The final result will be saved in the folder of the original images automatically. The names are the original couldy image name followed by 'cloud_remove' or 'cloud_remove_fast'


6. All the temporary files in the file folder will be cleared automatically when the process is finished.


7. The test data:
   1).all the Landsat images are preprocessed. They are ready for testing.

   2).500*500 size£¬DN value range is 0-255, in 2008.

   3).only have green (band1), red(band2), and NIR bands(band3),but the code can process image with more bands

   4).the parameters in the code have been set for this test data.
   
   5).computing time by PC: CLOUD_REMOVE_FAST.PRO is about 30 s, and CLOUD_REMOVE.PRO is about 3 minutes.