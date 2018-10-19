# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:53:53 2017

@author: CHENL
"""

import pandas as pd
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import numpy as np
import math
import os
import filePathInfor


# artificialdta
gridLeft = 0
gridBottom = 0

def proj_trans(lon, lat):
    p1 = Proj(init='epsg:4326')  # 地理坐标系WGS1984
    p2 = Proj(init='epsg:32650')  # 投影坐标WGS_1984_UTM_Zone_50N
    lon_val = lon.values
    lat_val = lat.values
    x1, y1 = p1(lon_val, lat_val)
    x2, y2 = transform(p1, p2, x1, y1, radians=True)
    return x2, y2

def grid_confirm(cellSize, gridNum, x, y):
    # 以网格左下角为原点，划分格网序号;
    xidArr = np.ceil((x - gridLeft) / cellSize);
    yidArr = np.ceil((y - gridBottom) / cellSize);
    outIndex = np.array([], dtype=np.bool)
    # x,y(1,gridNum = 30)
    for i in range(0, len(xidArr)):
        if (xidArr[i] < 1) | (xidArr[i] > gridNum):
            outIndex = np.append(outIndex, True)
        else:
            outIndex = np.append(outIndex, False)
    for j in range(0, len(yidArr)):
        if (yidArr[j] < 1) | (yidArr[j] > gridNum):
            outIndex[j] = True

    grid_id = (yidArr - 1) * gridNum + xidArr - 1
    # 标记出界点
    grid_id[outIndex] = -1
    # grid_id(0,gridNum*gridNum-1 = 899)
    totalNum = gridNum * gridNum - 1
    grid_id[(grid_id < 0) | (grid_id > totalNum)] = -1
    grid_id = grid_id.astype(np.int)
    return grid_id

def grid_inf(cellSize, gridNum):
    gridid_arr = np.arange(gridNum * gridNum)
    xid_arr = gridid_arr % gridNum
    yid_arr = gridid_arr // gridNum
    return gridid_arr, xid_arr, yid_arr  # 格网序列号（自然编码），格网行列号（地理编码）

def make_data(dataType):
    if dataType == 'single_nor':
        x_coor = 500
        y_coor = 500
        x = np.round(np.random.normal(x_coor , sigma , sampleNum),0)
        y = np.round(np.random.normal(y_coor , sigma , sampleNum),0)
    elif dataType == 'dual_nor':
        # sigma = 80
        # sampleNum = 400000
        x1 = np.round(np.random.normal(muXList[0] , sigma , sampleNum),0)
        y1 = np.round(np.random.normal(muYList[0],sigma,sampleNum),0)
        x2 = np.round(np.random.normal(muXList[1] , sigma , sampleNum),0)
        y2 = np.round(np.random.normal(muYList[1] , sigma , sampleNum),0)
        x = np.append(x1,x2)
        y = np.append(y1,y2)
    elif dataType == 'random':
        x = np.random.random(size = sampleNum)*1000
        y = np.random.random(size = sampleNum)*1000
    return x,y


# make dta
muXList = [300 , 700 , 24000]
muYList = [700 , 300 , 26000]

sigma = 150
sampleNum = 10000
simu = 1



# blockNum = 20
# blockCoe = int(1000/blockNum)  #50
# partIndex = []
# for i in range(0,blockNum+1):
#         partIndex.append(i*blockCoe)
#
# artDta = pd.DataFrame(index = range(1,1001,1),columns=range(1,1001,1))
# artDta.loc[:,:] = 0
#
# for i in range(0,20):
#     if i % 2 == 0:
#         for j in range(0,20,2):
#             artDta.iloc[partIndex[i]:partIndex[i+1],partIndex[j]:partIndex[j+1]] = 1
#     elif i%2 == 1:
#         for j in range(1,20,2):
#             artDta.iloc[partIndex[i]:partIndex[i+1], partIndex[j]:partIndex[j + 1]] = 1
# x = []
# y = []
# for i in range(1000):
#     for j in range(1000):
#         if artDta.iloc[i,j] == 1:
#             x.append(i+0.5)
#             y.append(j+0.5)
dtaType = 'single_nor'
for simu in range(11,15):
    x,y = make_data(dtaType)
    # # #指定不同的聚合尺度
    dta = pd.DataFrame(columns = ['x','y','checkin_num'])
    dta['x'] = x
    dta['y'] = y
    dta['checkin_num'] = 1

    odir = filePathInfor.get_filePath(dtaType, sigma, simu)
    if not os.path.exists(odir):
        os.mkdir(odir)
    dta.to_csv(odir + '/xy_dta.csv', index=None)

    # dta = pd.read_csv(odir + '/xy_dta.csv')
    # x = dta['x']
    # y = dta['y']


    for i,csize in enumerate(range(85,5,-5)):
        gridNum = int(1000/ csize)
        gridpos = grid_inf(csize,gridNum)
        df = pd.DataFrame()
        df['checkin_num'] = dta['checkin_num']
        #gridinf格网行列号信息
        gridinf = pd.DataFrame(index = gridpos[0],columns = ['gridx','gridy'])
        gridinf['gridx'] = gridpos[1]
        gridinf['gridy'] = gridpos[2]

        df['gridId'] = grid_confirm(csize,gridNum,x,y)
        df['num'] = 1
        checkin_cluster = df['checkin_num'].groupby(df['gridId']).sum()
        poi_cluster = df['num'].groupby(df['gridId']).sum()

        #sdd_df空间分布数据
        sdd_df = pd.concat([checkin_cluster,poi_cluster,gridinf],axis = 1).fillna(0)
        sdd_df.rename(columns = {'checkin_num':'checkin_sum','num':'poi_sum'},inplace = True)
        if (sdd_df.index[0] == -1):
            sdd_df.drop(sdd_df.index[0], inplace=True)  # 去掉网格id为-1的统计记录

        #绘制空间分布热力图
        checkin_arr = np.array(sdd_df['checkin_sum'])
        # np.savetxt(odir+'/checkinDta_'+str(csize)+'.csv',checkin_arr)


        checkin_matrix = checkin_arr.reshape(gridNum,gridNum)
        checkin_sort = sorted(sdd_df.checkin_sum , reverse = True)
        # norm = colors.Normalize(0,0)
        # fig,axs = plt.subplots(2,2)
        # axs = axs.flatten()
        # if i == 0:
        #     vmax= max(checkin_arr)
        #     vmin = 0
        #     norm = colors.Normalize(vmin,vmax)
        # ax = axs[i]
        # im = ax.imshow(checkin_matrix , plt.cm.Reds ,norm = norm)
        # ax.set_xticks([])
        # ax.set_yticks([])

        #lags,semivar

        oid_list = []
        did_list = []
        lags_list = []
        dsquraed_list = []
        count = 0
        start = time.time()
        for i in range(1,gridNum*gridNum):
                count += i
                o = gridinf.iloc[i]
                d_arr = gridinf.iloc[:i]
                lags = np.sqrt(pow((d_arr.gridx.tolist() - o.gridx), 2) + pow((d_arr.gridy.tolist() - o.gridy), 2)) * csize
                # 0列对应'checkin_sum'
                dif = abs(sdd_df.iloc[:i, 0] - sdd_df.iloc[i, 0])
                d_squared = pow(dif, 2)
                lags_list.extend(lags)
                dsquraed_list.extend(d_squared)

                if (count % 100000 == 0):
                    end = time.time()
                    print(count / 100000, 'timeconsum:', end - start)
                    start = time.time()

        vari_df = pd.DataFrame(columns = ['lags', 'vari','num'])
        vari_df['lags'] = lags_list
        vari_df['vari'] = dsquraed_list
        vari_df['num'] = 1
        vari_df.to_csv(odir + '/nor_varidf'+str(csize)+'.csv', index=None)


# plt.subplots_adjust(bottom = .05,top = 0.99,hspace =.002)
# fig.colorbar(im, ax=axs, fraction=.1)
# pos_cbar = fig.add_axes([0.95,0.2,0.075,0.6])
# cb=plt.colorbar(im,ax = axs[1])

# fig.tight_layout()
plt.show()

m = 1
