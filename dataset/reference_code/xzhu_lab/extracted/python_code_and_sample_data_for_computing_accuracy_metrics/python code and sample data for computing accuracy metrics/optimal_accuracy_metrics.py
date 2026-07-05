from osgeo import gdal, gdalconst
import numpy as np
import pandas as pd
import scipy.signal as ss
import tkinter as tk
from tkinter import filedialog
from datetime import datetime 
import os


def GetData(rasterFilePath):
    dataset=gdal.Open(rasterFilePath)
    rst_width = dataset.RasterXSize
    rst_Height = dataset.RasterYSize
    rst_Bands = dataset.RasterCount
    rst_geoTrans = dataset.GetGeoTransform()
    rst_proj =  dataset.GetProjection()
    
    #read raster
    ImgData= np.zeros((rst_Bands,rst_Height,rst_width))
    for bandIdx in range(0,rst_Bands):
        band = dataset.GetRasterBand(bandIdx + 1)
        ImgData[bandIdx,:,:] = band.ReadAsArray()
    return rst_width,rst_Height,rst_Bands\
        ,rst_geoTrans,rst_proj,ImgData
        
def Robert_edge(data):
    dataEdg = np.array(data-data)
    kernel1 = np.array([[1,0],
                        [0,-1]])
    kernel2 = np.array([[0,1],
                        [-1,0]])
    robertEdg = np.abs(ss.convolve(data,kernel1,mode = 'valid'))\
        +np.abs(ss.convolve(data,kernel2,mode = 'valid'))
    dataEdg[1:-1,1:-1] = robertEdg[1:,1:]
    return dataEdg

def LBP_original(data,tolerance):
    dis = data-data
    dis_temp = np.array([dis[1:-1,1:-1]]*8)
    for i in range(0,8):
        if i == 0:
            kernel = np.array([[0,0,0],
                               [1,0,0],
                               [0,0,0]])
        elif i == 1:
            kernel = np.array([[1,0,0],
                               [0,0,0],
                               [0,0,0]])
        elif i == 2:
            kernel = np.array([[0,1,0],
                               [0,0,0],
                               [0,0,0]])
        elif i == 3:
            kernel = np.array([[0,0,1],
                               [0,0,0],
                               [0,0,0]])
        elif i == 4:
            kernel = np.array([[0,0,0],
                               [0,0,1],
                               [0,0,0]])
        elif i == 5:
            kernel = np.array([[0,0,0],
                               [0,0,0],
                               [0,0,1]])
        elif i == 6:
            kernel = np.array([[0,0,0],
                               [0,0,0],
                               [0,1,0]])
        else:
            kernel = np.array([[0,0,0],
                               [0,0,0],
                               [1,0,0]])
        dis_temp[i,:,:] = ss.convolve(data,kernel,mode = 'valid')
        gtCenterIdx = np.round(dis_temp[i,::],8) > np.round((data[1:-1,1:-1] + tolerance),8)
        leCenterIdx = np.round(dis_temp[i,::],8) <= np.round((data[1:-1,1:-1] + tolerance),8)
        dis_temp[i,gtCenterIdx] = 1
        dis_temp[i,leCenterIdx] = 0
        
    dis_temp2 = dis_temp[0]*16 + dis_temp[1]*8 + dis_temp[2]*4 + dis_temp[3]*2 + dis_temp[4]*1 + dis_temp[5]*128 + dis_temp[6]*64 + dis_temp[7]*32 
    dis[1:-1,1:-1] = dis_temp2
    return dis


#main body
DN_min = 0.0      #minimal DN values of valid pixels
DN_max = 10000.0  #maximum DN values of valid pixels
tolerance = 0.005 #tolerance level for computing LBP feature,\
                  #which is the noise letter of images
                
#open reference image
FileName = filedialog.askopenfilename(title = 'open the reference fine-resolution image')
rstWidth,rstHeight,rstBands,rstGeoTrans,rstProj,rstData = \
    GetData(FileName)
true = rstData/DN_max
#true = rstData.astype(np.int16)

#open fused image
FileName0 = filedialog.askopenfilename(title = 'open the fused fine-resolution image')
rstWidth0,rstHeight0,rstBands0,rstGeoTrans0,rstProj0,rstData0 = \
    GetData(FileName0)
predict = rstData0/DN_max
#predict = rstData0.astype(np.int16)

#the initial time of program running
t0 = datetime.now()

v_result=np.zeros(4*(rstBands+1)).reshape(rstBands+1,4)

#compute RMSE and AD indices
for bandIdx in range(0,rstBands):
    x = np.array(true[bandIdx,:,:]).reshape(1,rstWidth*rstHeight).flatten()
    y = np.array(predict[bandIdx,:,:]).reshape(1,rstWidth*rstHeight).flatten()
    v_result[bandIdx,0] = np.mean(y-x)
    v_result[bandIdx,1] = np.sqrt(np.mean(np.square(y-x)))

reference=np.array(true)

# compute edge and LBP features for the reference image
# compute edge
edge_reference=np.array(reference)
for bandIdx in range(0,rstBands):
    predict_i=reference[bandIdx,:,:]
    edge_reference[bandIdx,:,:]=Robert_edge(predict_i)
    
#compute LBP Texture
lbp_reference = np.array(reference)
for bandIdx in range(0,rstBands):
    predict_i=reference[bandIdx,:,:]
    lbp_reference[bandIdx,:,:]=LBP_original(predict_i,tolerance)
    
#compute edge and LBP features for the fused image and compute the normalized difference
#compute edge
edge_predict=np.array(predict)
for bandIdx in range(0,rstBands):
    predict_i=predict[bandIdx,:,:]
    edge_predict[bandIdx,:,:]=Robert_edge(predict_i)
#compute the r of edge images
true_cutedge = np.array(edge_reference[:,1:-1,1:-1])
predict_cutedge = np.array(edge_predict[:,1:-1,1:-1])
for bandIdx in range(0,rstBands):
    x = np.array(true_cutedge[bandIdx,::]).reshape(1,(rstWidth-2)*(rstHeight-2)).flatten()
    #only use 0.9 quantile above pixels which represents edges in an image
    num_pure1=(rstWidth-2.0)*(rstHeight-2.0)
    sortIndex = np.argsort(x)
    #sortIndices = (np.linspace(0,int(num_pure1),int(num_pure1+1)))/num_pure1
    sortIndices = np.arange(0,(num_pure1+1))/num_pure1
    Percentiles=[0.9]
    dataIndices = np.array(np.where(sortIndices >= Percentiles)).flatten()
    data_90= x[sortIndex[dataIndices[0]]]
    #print,'90% edge:',data_90
    ind_realedge=np.array(np.where(x >= data_90)).flatten()
    x=np.array(x[ind_realedge])
    y=np.array(predict_cutedge[bandIdx,::]).reshape(1,(rstWidth-2)*(rstHeight-2)).flatten()
    y=np.array(y[ind_realedge])
    NDSI=(y-x)/(abs(y+x)+0.00001)
    v_result[bandIdx,2]=np.mean(NDSI)
    
#compute LBP Texture
lbp_predict=np.array(predict)
for bandIdx in range(0,rstBands):
    predict_i=np.array(predict[bandIdx,::])
    lbp_predict[bandIdx,::]=LBP_original(predict_i,tolerance)
predict_cutedge=np.array(lbp_predict[:,1:-1,1:-1])/255.0
true_cutedge=np.array(lbp_reference[:,1:-1,1:-1])/255.0
for bandIdx in range(0,rstBands):
    x = np.array(true_cutedge[bandIdx,::]).reshape(1,(rstWidth-2)*(rstHeight-2)).flatten()
    x = np.nan_to_num(x) #correct NAN
    y=np.array(predict_cutedge[bandIdx,::]).reshape(1,(rstWidth-2)*(rstHeight-2)).flatten()
    y = np.nan_to_num(y)  #correct NAN
    NDSI=(y-x)/(np.abs(y+x)+0.00001)
    v_result[bandIdx,3]=np.mean(NDSI)

#compute average of all bands for each index
for iind in range(0,4):
    v_result[rstBands,iind] = np.mean(v_result[0:rstBands,iind])
   
index = np.arange(1,rstBands+1)
indexname = ["band"+"%01d" % x for x in index]
indexname.append("bandAverage")
pd.DataFrame(v_result, index = indexname, columns = \
             ["AD","RMSE","EDGE","LBP"]).to_excel(os.path.dirname(FileName0) \
            +'\\'+os.path.basename(FileName0).split('.')[0]+'_accuracy.xlsx')
print('accuracy assessment results: AD, RMSE, EDGE, LBP')
np.set_printoptions(suppress=True)
print(v_result)

t1 = datetime.now()
timeConsume = 'time used:'+ str((t1-t0).seconds//3600)+' h '+\
    str((t1-t0).seconds%3600//60)+' m '+\
        str((t1-t0).seconds%60) + ' s'
print(timeConsume)