# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:35:15 2018

@author: SHOLIHIN RAHMAN
"""

import numpy as np
import cv2
from numpy import arctan2

#menghitung gradient
def gradient(image, same_size=False):

    sy, sx = image.shape
    if same_size:
        gx = np.zeros(image.shape)
        gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
        gx[:, 0] = -image[:, 0] + image[:, 1]
        gx[:, -1] = -image[:, -2] + image[:, -1]
    
        gy = np.zeros(image.shape)
        gy[1:-1, :] = image[:-2, :] - image[2:, :]
        gy[0, :] = image[0, :] - image[1, :]
        gy[-1, :] = image[-2, :] - image[-1, :]
    
    else:
        gx = np.zeros((sy-2, sx-2))
        gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

        gy = np.zeros((sy-2, sx-2))
        gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]
    
    return gx, gy

#menghitung magnitude dan orientasi
def magnitude_orientation(gx, gy):
       
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / np.pi) % 360
            
    return magnitude, orientation


def HOG(img, size_x, size_y):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist((np.power(gray/255., 0.25)*255).astype(np.uint8))
    resize = cv2.resize(gray, (size_x, size_y), interpolation = cv2.INTER_LINEAR)
    resize = np.float64(resize)
    y_, x_  = resize.shape
    gx,gy = gradient(resize,same_size=True) #menghitung nilai gradiet
    grad, ang = magnitude_orientation(gx, gy) #menghitung nilai magnitude
            
    sudut = np.zeros((y_,x_), np.float)
    piksel_sel = (8,8) #ukuran sel 8x8 piksel
    sel_blok = (2,2) #ukuran blok 2x2 sel atau 16x16 piksel
    cy, cx = piksel_sel
    cby, cbx = sel_blok
                
    #sel_x = x_//cx
    #sel_y = y_//cy
    p_over = (x_//cx) - cbx + 1
    l_over = (y_//cy) - cby + 1
            
    #total_blok = p_over * l_over
                
    #inisialisasi array fitur dengan ukuran 21 (banyak blok) x 36 (banyak fitur dalam satu blok)
    feature = np.zeros((p_over * l_over,36),dtype=float)
                
    mins = ang.min()
    maks = ang.max()
                
    #NORMALISASI SUDUT 0 - 180
    for m in range(y_):
        for n in range(x_):
            sudut[m,n] = ((ang[m,n]-mins)*(180-0))/((maks-mins)+mins)
                
    sudut = np.int64(sudut)
    grad = np.int64(grad)
                
    blk = 0
    #ITERASI SETIAP BLOK       
    for i in range(l_over):
        for j in range(p_over):
                                
            grad_blok = grad[cy*i:cy*i+(cy*cby),cx*j:cx*j+(cx*cbx)] 
            sudut_blok = sudut[cy*i:cy*i+(cy*cby),cx*j:cx*j+(cx*cbx)] 
                
            #blok_feature = np.zeros((4,9),dtype=float)
            blok_f1 = np.zeros((1,36),dtype=float)
                        
            #ITERASI SETIAP SEL PADA BLOK
            for a in range(cby):
                for b in range(cbx):
                                
                    grad_sel = grad_blok[cy*a:cy*a+cy , cx*b:cx*b+cx]
                    sudut_sel = sudut_blok[cy*a:cy*a+cy , cx*b:cx*b+cx]
                    bins = np.zeros((1,9),dtype=float)
                                
                    #ITERASI SETIAP PIKSEL DALAM SEL
                    for p in range(cy):
                        for q in range(cx):
                                        
                            alpha = sudut_sel[p,q]
                            gradien = grad_sel[p,q]
                        
                            if alpha > 10 and alpha <= 30:
                                bins[:,0] = bins[:,0] + gradien * (30 - alpha)//20
                                bins[:,1] = bins[:,1] + gradien * (alpha - 10)//20
                                            
                            elif alpha > 30 and alpha <= 50:
                                bins[:,1] = bins[:,1] + gradien * (50 - alpha)//20
                                bins[:,2] = bins[:,2] + gradien * (alpha - 30)//20
                                            
                            elif alpha > 50 and alpha <= 70:
                                bins[:,2] = bins[:,2] + gradien * (70 - alpha)//20
                                bins[:,3] = bins[:,3] + gradien * (alpha - 50)//20   
                                            
                            elif alpha > 70 and alpha <= 90:
                                bins[:,3] = bins[:,3] + gradien * (90 - alpha)//20
                                bins[:,4] = bins[:,4] + gradien * (alpha - 70)//20
                                            
                            elif alpha > 90 and alpha <= 110:
                                bins[:,4] = bins[:,4] + gradien * (110 - alpha)//20
                                bins[:,5] = bins[:,5] + gradien * (alpha - 90)//20
                                            
                            elif alpha > 110 and alpha <= 130:
                                bins[:,5] = bins[:,5] + gradien * (130 - alpha)//20
                                bins[:,6] = bins[:,6] + gradien * (alpha - 110)//20
                                            
                            elif alpha > 130 and alpha <= 150:
                                bins[:,6] = bins[:,6] + gradien * (150 - alpha)//20
                                bins[:,7] = bins[:,7] + gradien * (alpha - 130)//20
                                            
                            elif alpha > 150 and alpha <= 170:
                                bins[:,7] = bins[:,7] + gradien * (170 - alpha)//20
                                bins[:,8] = bins[:,8] + gradien * (alpha - 150)//20
                                            
                            elif alpha >= 0 and alpha <= 10:
                                bins[:,0] = bins[:,0] + gradien * (alpha + 10)//20
                                bins[:,8] = bins[:,8] + gradien * (10 - alpha)//20
                                            
                            elif alpha > 170 and alpha <= 180:
                                bins[:,8] = bins[:,8] + gradien * (190 - alpha)//20
                                bins[:,0] = bins[:,0] + gradien * (alpha - 170)//20 
                                
                                
                    blok_f1 [:,9*(a*2+b):9*(a*2+b+1)] = bins
                                
            blok_f1 = blok_f1 / ((abs(blok_f1)**2 + (0.01)**2)**0.5) #normalisasi fitur blok
            feature [blk,:] = blok_f1 #menggabungkan fitur setiap blok
            blk = blk + 1
                        
                                
    fitur = np.reshape(feature, (1,756))
    return fitur, resize
