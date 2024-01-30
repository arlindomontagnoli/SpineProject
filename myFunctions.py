import cv2
import numpy as np,sys
import os
import pickle
import math
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
from pydicom import dcmread
from scipy.interpolate import UnivariateSpline
from screeninfo import get_monitors
import csv
from scipy.interpolate import splprep,splev
from random import seed
from random import randrange
import pandas as pd
import copy

#-------------------------------------------------------------------
# RotatePoint
#-------------------------------------------------------------------
def RotatePoint(centro, angle, p):
    s = np.sin(angle)
    c = np.cos(angle)
    # translate point back to origin:
    px =p[0] - centro[0]
    py =p[1] - centro[1]

    # rotate point
    xnew = px * c - py * s
    ynew = px * s + py * c

    # translate point back:
    
    ptx = xnew + centro[0]
    pty = ynew + centro[1]
    return ptx,pty

#-------------------------------------------------------------------
# Curvature
#-------------------------------------------------------------------
def Curvature(x,y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = (d2x * dy - dx * d2y) / (dx * dx + dy * dy)**1.5
    return curvature
#-------------------------------------------------------------------
# Normal
#-------------------------------------------------------------------
def Normal(x,y):
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    nx = dy / np.sqrt(dx * dx + dy * dy)
    ny = -dx /np.sqrt(dx * dx + dy * dy)
    return nx,ny
#-------------------------------------------------------------------
# Slope
#-------------------------------------------------------------------
def Slope(x,y):
    if(x==0):
        if(y>0): a=np.pi/2
        if(y<0): a=-np.pi/2
        if(y==0):a=0
    else:
        a = np.arctan( y/x )
    if (x<0):
        if (y >=0): a += np.pi
        if (y <  0): a -= np.pi
    return a
#-------------------------------------------------------------------
# Angle
#-------------------------------------------------------------------
def Angle(pt0,pt1):
    x=pt1[0]-pt0[0]
    y=pt0[1]-pt1[1]
    a = Slope(x,y)
    return a
#-------------------------------------------------------------------
# det
#-------------------------------------------------------------------
def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
#-------------------------------------------------------------------
# LineIntersection
#-------------------------------------------------------------------
def LineIntersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       #raise Exception('lines do not intersect')
       return 0, 0

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
#-------------------------------------------------------------------
# Distance
#-------------------------------------------------------------------
def Distance(pt0,pt1):
    d = np.sqrt((pt1[0]-pt0[0])**2 +(pt1[1]-pt0[1])**2)
    return d
#-------------------------------------------------------------------
# MinDistance
#-------------------------------------------------------------------
def MinDistance(pt0,pts):
    minDist = 10000000
    for i in range(len(pts)):
        d = Distance(pt0,pts[i])
        if d < minDist and d>=0:
            index  = i
            minDist = d
    return index,minDist
    
#-------------------------------------------------------------------
# SortDistance
#-------------------------------------------------------------------
def SortDistance(pt0,pts):
    
    index = list(range(len(pts)))
    minDistAux = 1
    for i in range(len(pts)):
        minDist = 10000000
        for j in range(len(pts)):
            d = Distance(pt0,pts[j])
            if d < minDist and d > minDistAux and d>1:
                index[i]=j
                minDist = d
        minDistAux = minDist
    return index
#-------------------------------------------------------------------
# AngleBetweenLines
#-------------------------------------------------------------------
def AngleBetweenLines(line1,line2):
    co0 = line1[1][0]-line1[0][0]
    co1 = line2[1][0]-line2[0][0]
    if co0 != 0 and co1!=0:
        m1= (line1[0][1]-line1[1][1])/co0 #slope of line1 -y   
        m2= (line2[0][1]-line2[1][1])/co1 #slope of line2 -y
        if((1+m1*m2)==0):
            a = np.pi/2
        else:
            k = (m1-m2)/(1+m1*m2)
            a = np.arctan(k)
    elif co0!= 0:
        m1= (line1[0][1]-line1[1][1])/co0 #slope of line1 -y   
        a = math.pi/2 - np.arctan(m1)
    elif co1!=0: 
        m2= (line2[0][1]-line2[1][1])/co1 #slope of line2 -y
        a = math.pi/2 - np.arctan(m2)
    else:
        a= 0  
    return  a 
#-------------------------------------------------------------------
# Femur
#-------------------------------------------------------------------
def Femur(vertices):
    maxY = 0
    idx0 = 0
    aux =  vertices.copy()
    for i in range(len(vertices)):
        if(vertices[i][1]>maxY):
            maxY = vertices[i][1]
            idx0 = i
    index = SortDistance(vertices[idx0],vertices)
    return [vertices[idx0],vertices[index[0]],vertices[index[1]],vertices[index[2]]]

#-------------------------------------------------------------------
# Sacrum
#-------------------------------------------------------------------
def Sacrum(vertices,femur):
    vS1=[[0,0],[0,0],[0,0],[0,0]]

    maiorY = 0
    for i in range(len(vertices)):
        if vertices[i][1]>vertices[maiorY][1]: maiorY=i
    mY = vertices[maiorY]
    ##############################
    
    index = SortDistance(vertices[maiorY],vertices)
    pts2= [vertices[index[0]],vertices[index[1]],vertices[index[2]],vertices[index[3]],vertices[index[4]]]#sem maior y
    pts=pts2
    pts.append(vertices[maiorY])

    flagRight = True######False!!!
    if(vertices[maiorY][0]<femur[0]): flagRight=True  
    
    if(flagRight):
        vS1[1]=vertices[maiorY]


        minx=1000000
        for i in range (5):  #menor x
            if(pts2[i][0]<minx):
                minx=pts2[i][0]
                idx = i
        pts3=[]
        for i in range (5):#remove o menor x
            if i!=idx:
                pts3.append(pts2[i])
            else:
                vS1[0] = pts2[i]
        
        c = [(vS1[0][0] + vS1[1][0])/2, (vS1[0][1] + vS1[1][1])/2]


        vS1=Vertebra(pts,c,'S1 - L5')


  
        
    return(vS1,flagRight)
    

    
#-------------------------------------------------------------------
# Order
#-------------------------------------------------------------------
def Order(pts,cb,vname):
    cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) /4
    cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) /4
    ptL=[]
    ptR=[]
    aref = Angle(cb,(cx,cy))
    
    for i in range (4):
        ang = Angle(cb,pts[i])
        if(ang<aref):
            ptL.append(pts[i])
        else:
            ptR.append(pts[i])
    if(len(ptL)<2 or len(ptR)<2):
       # print("aref:",aref*180/3.1415)
       # for i in range (4):
       #     ang = Angle(cb,pts[i])
       #      print("ang:",ang*180/3.1415)
        print('===========================================================')
        print('    ERRO: Favor conferir a região da vértebra '+vname)
        print('===========================================================')
    
    V=[ptL[0],ptR[0],ptL[1],ptR[1]]    
    '''
    if(Distance(cb,ptL[1])<Distance(cb,ptL[0])):
        V[1] = ptL[1]
        V[3] = ptL[0]
    if(Distance(cb,ptR[1])<Distance(cb,ptR[0])):
        V[0] = ptR[1]
        V[2] = ptR[0]    
    '''    
    
    return V
    

#-------------------------------------------------------------------
# Vertebra
#-------------------------------------------------------------------
def Vertebra(vertices,center,vname):
    index = SortDistance(center,vertices)
    pts = [vertices[index[0]],vertices[index[1]],vertices[index[2]],vertices[index[3]],vertices[index[4]],vertices[index[5]]]

    menorX1 = 2  #menor x nos 4 ultimos (Esquerda)
    for i in range(3,6):
        if pts[menorX1][0]>pts[i][0]: menorX1=i
    menorX2 = 2  #2o menor x nos 4 ultimos
    for i in range(3,6):
        if i!=menorX1:
            if pts[menorX2][0]>pts[i][0]: menorX2=i    
    maiorYE = menorX1 
    if pts[menorX2][1]>pts[menorX1][1]: maiorYE = menorX2

    maiorX1 = 2  #maior x nos 4 ultimos (Dir)
    for i in range(3,6):
        if pts[maiorX1][0]<pts[i][0]: maiorX1=i
    maiorX2 = 2  #2o m x nos 4 ultimos
    for i in range(3,6):
        if i!=maiorX1:
            if pts[maiorX2][0]<pts[i][0]: maiorX2=i    
    maiorYD = maiorX1 
    if pts[maiorX2][1]>pts[maiorX1][1]: maiorYD = maiorX2
            

    vo = Order([pts[0],pts[1],pts[maiorYE],pts[maiorYD]],center,vname)
    
    '''a0 = AngleBetweenLines((pts[0],pts[1]),(pts[0],pts[2]))
    a1 = AngleBetweenLines((pts[0],pts[1]),(pts[0],pts[3]))
    a2 = AngleBetweenLines((pts[0],pts[1]),(pts[0],pts[4]))
    a3 = AngleBetweenLines((pts[0],pts[1]),(pts[0],pts[5]))
    a4 = AngleBetweenLines((pts[0],pts[2]),(pts[0],pts[3]))
    a5 = AngleBetweenLines((pts[0],pts[2]),(pts[0],pts[4]))
    a6 = AngleBetweenLines((pts[0],pts[2]),(pts[0],pts[5]))
    a7 = AngleBetweenLines((pts[0],pts[3]),(pts[0],pts[4]))
    a8 = AngleBetweenLines((pts[0],pts[3]),(pts[0],pts[5]))
    a9 = AngleBetweenLines((pts[0],pts[4]),(pts[0],pts[5]))
    r = math.pi/2
    i=[[0,1,2],[0,1,3],[0,1,4],[0,1,5],[0,2,3],[0,2,4],[0,2,5],[0,3,4],[0,3,5],[0,4,5]]
    av = [abs(abs(a0)-r),abs(abs(a1)-r),abs(abs(a2)-r),abs(abs(a3)-r),abs(abs(a4)-r),abs(abs(a5)-r),abs(abs(a6)-r),abs(abs(a7)-r),abs(abs(a8)-r),abs(abs(a9)-r)]
    ias = sorted(range(len(av)), key=lambda k: av[k])  #indice angulo retos ordenados
    ic = i[ias[0]]+i[ias[1]] #ndice dos dois mais retos
    dp = set([x for x in ic if ic.count(x) > 1]) #duplicados
    ics = set(ic)
    ndp = list(ics - dp)  
    lessy=ndp[0]  #menor y
    if(pts[ndp[0]][1]<pts[ndp[1]][1]): lessy=ndp[1]   
    s = set([0,1,2,3,4,5])
    ri = list(s-ics)
    prem = list(ri)
    lessy2=prem[0]
    
    if(pts[prem[0]][1]<pts[prem[1]][1]): lessy2=prem[1]
    dpl = list(dp)
    iv = [round(dpl[0]),round(dpl[1]),round(lessy),round(lessy2)]
       
    vo = Order([pts[iv[0]],pts[iv[1]],pts[iv[2]],pts[iv[3]]],center)
    '''
    
   
    return vo
    
    
#-------------------------------------------------------------------
# Assert
#------------------------------------------------------------------- 
def Assert(vertices):
    for i in range(len(vertices)):
        for j in range(len(vertices)):
            vertices[i][0]=vertices[i][0]
            vertices[i][0]=vertices[i][0]

            d = Distance(vertices[i],vertices[j])
            if i!=j:
                if d<0.2:
                    print(vertices[i])
                    print(vertices[j])
                    print('Duplicidade?')
                    
                '''           
                if(d<=1 and i!=j):
                    if(vertices[i][0]<=vertices[j][0]):
                        vertices[i][0]=vertices[i][0]-1
                    else:
                        vertices[i][0]=vertices[i][0]+1
                    if(vertices[i][1]<=vertices[j][1]):
                        vertices[i][1]=vertices[i][1]-1
                    else:
                        vertices[i][1]=vertices[i][1]+1    
                '''
#-------------------------------------------------------------------
# VRand
#------------------------------------------------------------------- 
def VRand(vertices,sd):
    seed(sd+10)
    for i in range(len(vertices)):
        r = randrange(-3,3)
        vertices[i][0]=vertices[i][0]+r
        r = randrange(-3,3)
        vertices[i][1]=vertices[i][1]+r
       
#-------------------------------------------------------------------
# Draw Circle
#-------------------------------------------------------------------
def Circle(image,center,radio,color,width):
    cv2.circle(image,(round(center[0]),round(center[1])),radio,color,width)
#-------------------------------------------------------------------
# Draw Line
#------------------------------------------------------------------- 
def Line(image,pt1,pt2,color,width):    
    cv2.line(image,(round(pt1[0]),round(pt1[1])),(round(pt2[0]),round(pt2[1])),color, width)

    