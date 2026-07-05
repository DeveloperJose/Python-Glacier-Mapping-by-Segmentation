;---------------------------------------------------------------------------
;                       FILL THE SLC-OFF GAP OF ETM+
;                    Using multiple SLC-off ETM+ images
;     VERSION 2: can be used for whole ETM scene,modify the bug in version 1
;            Developed by Zhu Xiaolin,email: zhuxiaolin.rs@gmail.com
;             Beijing Normal University,Ohio State University
;                    Dr.Jin Chen, email:chenjin@ires.cn
;                       Beijing Normal University
;                                  2010-8-1
;                      Copyright belong to Chen's Lab
;---------------------------------------------------------------------------

1. The program is written by IDL;

2. Before run the program, ENVI should be opened first because the program use some functions of ENVI£»

3. Parameters for FILLGAP_SINGLE_V1.pro
(1) min_similar=20          
Set the minimum sample size of similar pixels, 20 was used in my test. 20-50 was recommended;
(2)max_window=8          
Set the maximum window size (half), 8 is same with USGS¡¯ method;
(3) num_class=4             
Set the estimated number of classes according to the scene; 
(4)num_input=2                    
Set the number of input images
(5) DN_min=0  
    DN_max=255             
Set the range of DN value of the image. If it is byte, the range is from 0 to 255.
(6)patch_long=1000                 
set the size of block,which is determined by the size of image. If process whole ETM scene, set 1000
(7)temp_file='D:\temp'            
Set the temporary file address. Please build a file named "temp" before run the program, and write the address of this file in program, such as 'D:\temp'.
  

4. Output the filled result

Both the filled result and mask image will be saved in the file of the original ETM+ image automatically. The names are the original ETM+ image name followed by 'filled'and 'mask'
such as: 20080922ETM_filled and 20080922ETM_mask


5. All the temporary data in the temporary file will be cleared automatically when the process is finished.


Note: when using FILLGAP_MULTIPLE.pro, the input images should be opened one by one according to their ranks. The nearest image is opened first.
