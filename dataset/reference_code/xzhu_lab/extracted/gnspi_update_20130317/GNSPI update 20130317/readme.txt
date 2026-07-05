;---------------------------------------------------------------------------
;           GNSPI algorithm for FILLING THE SLC-OFF GAP OF ETM+ IMAGES
;                           Using TM input images
;        Can process whole ETM+ scene using block strategy
;            Developed by Xiaolin Zhu,email: zhu.381@osu.edu
;             Department of Geography,The Ohio State University                  
;                         update date:  2012-5-5
;                     Copyright belong to Xiaolin Zhu
; Please cite the reference:
; Zhu, X., Liu, D. and Chen, J. 2012. A new geostatistical approach for filling
;      gaps in Landsat ETM+ SLC-off images, Remote Sensing of Environment,
;      124,49-60. 
;---------------------------------------------------------------------------

1. The program is written by IDL;

2. Before running the program, ENVI should be opened first because the program uses some functions of ENVI£»

3. Parameters for GNSPI.pro
;----------------------------------------------------------------------
 sample_size=20                 ;set the sample size of sample pixels
 size_wind=12                   ;set the maximum window size
 class_num=4                    ;set the estimated number of classes
 num_series=1                   ;set the number of images in the time-series except the input image
 DN_min=0.0                       ;set the range of DN value of the image,If byte, 0 and 255
 DN_max=1.0
 patch_long=500                ;set the size of block,if process whole ETM scene, set 1000
 temp_file='G:\temp'            ;set the temporary file location
;------------------------------------------------------------------------
Go to the main program part the set these parameters

(1)sample_size=20 
 the number of samples, 20 is recommended
(2)size_wind=12         
Set the window size (half),  if 12, the window size is 12*2+1=25;
(3) num_class=4             
Set the estimated number of classes according to the scene; 
(4) num_series=1                   
set the number of images in the time-series except the input image, 1 means there are two images will be used to selet the samples
(5) DN_min=0  
    DN_max=1             
Set the range of DN value of the image. If it is byte, the range is from 0 to 255.
(6)patch_long=500                 
set the size of block,which is determined by the size of image. If process the whole TM scene, 500 is recommended.(block  is to solve the problem of computer memory limit).
(7)temp_file='D:\temp'            
Set the temporary file address. Please build a folder named "temp" before run the program, and write the address of this folder in program, such as 'D:\temp'.
  
4. Input the images accoring to the name of window

5. Output the filled result

The filled result will be saved in the folder of the original images automatically. The names are the original gap image name followed by '_filled_GNSPI'

The uncertainty file is also saved in the folder.


6. All the temporary data in the temporary file will be cleared automatically when the process is finished.


7. The test data:
   1).all the images are preprocessed. They are ready for GNSPI.

   2).500*500 size£¬value range is 0-1, in 2010.

   3).only have green (band1), red(band2), and NIR bands(band3),but the code can process image with more bands
   4). the parameters in the code have been set for this test data.