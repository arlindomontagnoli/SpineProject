
# -*- coding: utf-8 -*-

from myFunctions import *
import pandas as pd
#-------------------------------------------------------------------
# main
#-------------------------------------------------------------------
def main():

    global vertices,imgCrop,imgCrop0,roi,scl
   
    Tk().withdraw()
    
    os.system('cls||clear')
    screen = get_monitors()[0]
    Tk().withdraw()
    file = askopenfilename(initialfile = "File.dcm", title = "Select Dicom or Image file", filetypes = (("image File",(".dcm",".png",".jpg",".jpeg")),("all files",".*")))
    dirName = os.path.dirname(file)
    fileName = os.path.basename(file)
    fname, fextension = os.path.splitext(fileName)
    flagRef = True
    if(fextension=='.dcm'):
        ds = dcmread(file, force=True)
        height,width = ds.pixel_array.shape
        image = np.uint8(ds.pixel_array*0.07)
        scl = max(width/screen.width,height/screen.height) +0.05
    else:
        image = cv2.imread(file)
        height,width,channels = image.shape
        scl = max(width/screen.width,height/screen.height) +0.05
    try:
        open_file = open(dirName+'/'+fileName + '_reference.pkl', "rb")
        reference = pickle.load(open_file)
        open_file.close()
        
    except:
        reference = [[0,0],[0,10]]; 
        flagRef = False
    try:
        open_file = open(dirName+'/'+fileName + '_vertices.pkl', "rb")
        vertices = pickle.load(open_file)
        open_file.close()
        
    except:
        vertices=[]
    origVertices =  vertices.copy()
    
    c=0
     
    metricRef = 10/Distance(reference[0],reference[1])
    #print('Reference:',metricRef,reference[0],reference[1])
    Assert(vertices)
    vF = Femur(vertices)
    for v in vF:
        vertices.remove(v)
       
    cFemurx = (vF[0][0] + vF[1][0] + vF[2][0] + vF[3][0]) /4
    cFemury = (vF[0][1] + vF[1][1] + vF[2][1] + vF[3][1]) /4
    cFemur =[cFemurx,cFemury]
    i,fr = MinDistance(cFemur,vF)
    Circle(image,(cFemurx,cFemury),2,(50,250,20),-1)
    Circle(image,(cFemurx,cFemury),round(fr),(50,250,20),1)



    ordVertices=[]
 
    vS1,flagRight = Sacrum(vertices,cFemur)
    ordVertices.append(vS1)

    for v in vS1:
        vertices.remove(v)
        Circle(image,(v),1,(50,250,250),-1)
     

        
    cS1x = (vS1[0][0] + vS1[1][0] + vS1[2][0] + vS1[3][0]) /4
    cS1y = (vS1[0][1] + vS1[1][1] + vS1[2][1] + vS1[3][1]) /4
    cS1 =[cS1x,cS1y]
    cS1tx = (vS1[2][0] + vS1[3][0]) /2
    cS1ty = (vS1[2][1] + vS1[3][1]) /2
    cS1t =[cS1tx,cS1ty]
    
    cS1bx = (vS1[0][0] + vS1[1][0]) /2
    cS1by = (vS1[0][1] + vS1[1][1]) /2
    if cS1by < cS1y:
        cS1by=cS1y+0.01
    cS1b =[cS1bx,cS1by]
    Circle(image,(cS1),2,(50,150,250),-1)
    #Circle(image,(cS1b),2,(250,240,40),-1)
    vL5 = Vertebra(vertices,cS1t,'S1 - L5')
    
    ordVertices.append(vL5)
    c=0
    for v in vL5:
        vertices.remove(v)
        Circle(image,(v),1,(250,50,50),-1)
    

  


    cL5x = (vL5[0][0] + vL5[1][0] + vL5[2][0] + vL5[3][0]) /4
    cL5y = (vL5[0][1] + vL5[1][1] + vL5[2][1] + vL5[3][1]) /4
    cL5 =[cL5x,cL5y]
    
    vL4 = Vertebra(vertices,cL5,'L5 - S1 ')
    ordVertices.append(vL4)
    for v in vL4:
        vertices.remove(v)
        Circle(image,(v),1,(250,50,50),-1)
    cL4x = (vL4[0][0] + vL4[1][0] + vL4[2][0] + vL4[3][0]) /4
    cL4y = (vL4[0][1] + vL4[1][1] + vL4[2][1] + vL4[3][1]) /4
    cL4 =[cL4x,cL4y]
    
    
    vL3 = Vertebra(vertices,cL4,'L4 - L5')
    ordVertices.append(vL3)
    for v in vL3:
        vertices.remove(v)
        Circle(image,(v),1,(250,50,50),-1)
    cL3x = (vL3[0][0] + vL3[1][0] + vL3[2][0] + vL3[3][0]) /4
    cL3y = (vL3[0][1] + vL3[1][1] + vL3[2][1] + vL3[3][1]) /4
    cL3 =[cL3x,cL3y]
    #resizedImg =  cv2.resize(image,(int(width/1),(int(height/1))))
    #cv2.imshow('Image', resizedImg)
    #x =input()
    vL2 = Vertebra(vertices,cL3,'L3 - L4')
    ordVertices.append(vL2)
    for v in vL2:
        vertices.remove(v)
        Circle(image,(v),1,(250,50,50),-1)
    cL2x = (vL2[0][0] + vL2[1][0] + vL2[2][0] + vL2[3][0]) /4
    cL2y = (vL2[0][1] + vL2[1][1] + vL2[2][1] + vL2[3][1]) /4
    cL2 =[cL2x,cL2y]
    #cv2.imshow('Image', image)
    #key=cv2.waitKey(0)
    vL1 = Vertebra(vertices,cL2,'L2 - L3')
    ordVertices.append(vL1)
    for v in vL1:
        vertices.remove(v)
        Circle(image,(v),1,(250,50,50),-1)
    cL1x = (vL1[0][0] + vL1[1][0] + vL1[2][0] + vL1[3][0]) /4
    cL1y = (vL1[0][1] + vL1[1][1] + vL1[2][1] + vL1[3][1]) /4
    cL1 =[cL1x,cL1y]
    
    vT12 = Vertebra(vertices,cL1,'L1 - L2')
    ordVertices.append(vT12)
    for v in vT12:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT12x = (vT12[0][0] + vT12[1][0] + vT12[2][0] + vT12[3][0]) /4
    cT12y = (vT12[0][1] + vT12[1][1] + vT12[2][1] + vT12[3][1]) /4
    cT12 =[cT12x,cT12y]
    
    vT11 = Vertebra(vertices,cT12,'T12 -L1')
    ordVertices.append(vT11)
    for v in vT11:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT11x = (vT11[0][0] + vT11[1][0] + vT11[2][0] + vT11[3][0]) /4
    cT11y = (vT11[0][1] + vT11[1][1] + vT11[2][1] + vT11[3][1]) /4
    cT11 =[cT11x,cT11y]
    
    vT10 = Vertebra(vertices,cT11,'T11 - T12')
    ordVertices.append(vT10)
    for v in vT10:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT10x = (vT10[0][0] + vT10[1][0] + vT10[2][0] + vT10[3][0]) /4
    cT10y = (vT10[0][1] + vT10[1][1] + vT10[2][1] + vT10[3][1]) /4
    cT10 =[cT10x,cT10y]
    
    vT9 = Vertebra(vertices,cT10,'T10 - T11')
    ordVertices.append(vT9)
    for v in vT9:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT9x = (vT9[0][0] + vT9[1][0] + vT9[2][0] + vT9[3][0]) /4
    cT9y = (vT9[0][1] + vT9[1][1] + vT9[2][1] + vT9[3][1]) /4
    cT9 =[cT9x,cT9y]
    
    vT8 = Vertebra(vertices,cT9,'T9 - T10')
    ordVertices.append(vT8)
    for v in vT8:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT8x = (vT8[0][0] + vT8[1][0] + vT8[2][0] + vT8[3][0]) /4
    cT8y = (vT8[0][1] + vT8[1][1] + vT8[2][1] + vT8[3][1]) /4
    cT8 =[cT8x,cT8y]

    vT7 = Vertebra(vertices,cT8,'T8 - T9')
    ordVertices.append(vT7)
    for v in vT7:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT7x = (vT7[0][0] + vT7[1][0] + vT7[2][0] + vT7[3][0]) /4
    cT7y = (vT7[0][1] + vT7[1][1] + vT7[2][1] + vT7[3][1]) /4
    cT7 =[cT7x,cT7y]
    
    vT6 = Vertebra(vertices,cT7,'T7 - T8')
    ordVertices.append(vT6)
    for v in vT6:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT6x = (vT6[0][0] + vT6[1][0] + vT6[2][0] + vT6[3][0]) /4
    cT6y = (vT6[0][1] + vT6[1][1] + vT6[2][1] + vT6[3][1]) /4
    cT6 =[cT6x,cT6y]
    
    vT5 = Vertebra(vertices,cT6,'T6 - T7')
    ordVertices.append(vT5)
    for v in vT5:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT5x = (vT5[0][0] + vT5[1][0] + vT5[2][0] + vT5[3][0]) /4
    cT5y = (vT5[0][1] + vT5[1][1] + vT5[2][1] + vT5[3][1]) /4
    cT5 =[cT5x,cT5y]
    
    
    vT4 = Vertebra(vertices,cT5,'T5 - T6')
    ordVertices.append(vT4)
    i=0
    for v in vT4:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT4x = (vT4[0][0] + vT4[1][0] + vT4[2][0] + vT4[3][0]) /4
    cT4y = (vT4[0][1] + vT4[1][1] + vT4[2][1] + vT4[3][1]) /4
    cT4 =[cT4x,cT4y]
    #cv2.imshow('Image', image)
    #key=cv2.waitKey(0)
   
    vT3 = Vertebra(vertices,cT4,'T4 - T5') 
    ordVertices.append(vT3)
    for v in vT3:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT3x = (vT3[0][0] + vT3[1][0] + vT3[2][0] + vT3[3][0]) /4
    cT3y = (vT3[0][1] + vT3[1][1] + vT3[2][1] + vT3[3][1]) /4
    cT3 =[cT3x,cT3y]
    
    vT2 = Vertebra(vertices,cT3,'T3 - T4')
    ordVertices.append(vT2)
    for v in vT2:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT2x = (vT2[0][0] + vT2[1][0] + vT2[2][0] + vT2[3][0]) /4
    cT2y = (vT2[0][1] + vT2[1][1] + vT2[2][1] + vT2[3][1]) /4
    cT2 =[cT2x,cT2y]
    
    vT1 = Vertebra(vertices,cT2,'T2 - T3')
    ordVertices.append(vT1)
    for v in vT1:
        vertices.remove(v)
        Circle(image,(v),1,(10,80,255),-1)
    cT1x = (vT1[0][0] + vT1[1][0] + vT1[2][0] + vT1[3][0]) /4
    cT1y = (vT1[0][1] + vT1[1][1] + vT1[2][1] + vT1[3][1]) /4
    cT1 =[cT1x,cT1y]
    
   
    

    vC7 = Vertebra(vertices,cT1,'C7 - T1 - T2')
    ordVertices.append(vC7)
    for v in vC7:
        vertices.remove(v)
        Circle(image,(v),1,(50,240,40),-1)
    cC7x = (vC7[0][0] + vC7[1][0] + vC7[2][0] + vC7[3][0]) /4
    cC7y = (vC7[0][1] + vC7[1][1] + vC7[2][1] + vC7[3][1]) /4
    cC7 =[cC7x,cC7y]

    #cv2.imshow('Image', image)
    #key=cv2.waitKey(0)


    vC6 = Vertebra(vertices,cC7,'C6 - C7 - T1')
    ordVertices.append(vC6)
    for v in vC6:
        vertices.remove(v)
        Circle(image,(v),1,(50,240,40),-1)
    cC6x = (vC6[0][0] + vC6[1][0] + vC6[2][0] + vC6[3][0]) /4
    cC6y = (vC6[0][1] + vC6[1][1] + vC6[2][1] + vC6[3][1]) /4
    cC6 =[cC6x,cC6y]



    vC5 = Vertebra(vertices,cC6,'C5 - C6 - C7')
    ordVertices.append(vC5)
    for v in vC5:
        vertices.remove(v)
        Circle(image,(v),1,(50,240,40),-1)
    cC5x = (vC5[0][0] + vC5[1][0] + vC5[2][0] + vC5[3][0]) /4
    cC5y = (vC5[0][1] + vC5[1][1] + vC5[2][1] + vC5[3][1]) /4
    cC5 =[cC5x,cC5y]
    #cv2.imshow('Image', image)
    #key=cv2.waitKey(0)
    vC4 = Vertebra(vertices,cC5,'C4 - C5 - C6')
    ordVertices.append(vC4)
    for v in vC4:
        vertices.remove(v)
        Circle(image,(v),1,(50,240,40),-1)
    cC4x = (vC4[0][0] + vC4[1][0] + vC4[2][0] + vC4[3][0]) /4
    cC4y = (vC4[0][1] + vC4[1][1] + vC4[2][1] + vC4[3][1]) /4
    cC4 =[cC4x,cC4y]
    #cv2.imshow('Image', image)
    #key=cv2.waitKey(0)
    vC3 = Vertebra(vertices,cC4,'C3 - C4 - C5')
    ordVertices.append(vC3)
    for v in vC3:
        vertices.remove(v)
        Circle(image,(v),1,(50,240,40),-1)
    cC3x = (vC3[0][0] + vC3[1][0] + vC3[2][0] + vC3[3][0]) /4
    cC3y = (vC3[0][1] + vC3[1][1] + vC3[2][1] + vC3[3][1]) /4
    cC3 =[cC3x,cC3y]
    #cv2.imshow('Image', image)
    #key=cv2.waitKey(0)
    vC2 = vertices#Vertebra(vertices,cC3,'C2 - C3 - C4')
    '''
    ordVertices.append(vC2)
    cC2x = (vC2[0][0] + vC2[1][0] + vC2[2][0] + vC2[3][0]) /4
    cC2y = (vC2[0][1] + vC2[1][1] + vC2[2][1] + vC2[3][1]) /4
    cC2 =[cC2x,cC2y]
    for v in vC2:
        #vertices.remove(v)
        Circle(image,(v),1,(50,240,40),-1)
    '''
    df = pd.DataFrame(ordVertices)
    df.to_csv(dirName+'\ordVertices.csv', encoding='utf-8', index=False)
    
    
    
    cxs=[]
    cys=[]
    #cxs.append(cC2x)
    #cys.append(cC2y)
    cxs.append(cC3x)
    cys.append(cC3y)
    cxs.append(cC4x)
    cys.append(cC4y)
    cxs.append(cC5x)
    cys.append(cC5y)
    cxs.append(cC6x)
    cys.append(cC6y)
    cxs.append(cC7x)
    cys.append(cC7y)
    cxs.append(cT1x)
    cys.append(cT1y)
    cxs.append(cT2x)
    cys.append(cT2y)
    cxs.append(cT3x)
    cys.append(cT3y)
    cxs.append(cT4x)
    cys.append(cT4y)
    cxs.append(cT5x)
    cys.append(cT5y)
    cxs.append(cT6x)
    cys.append(cT6y)
    cxs.append(cT7x)
    cys.append(cT7y)
    cxs.append(cT8x)
    cys.append(cT8y)
    cxs.append(cT9x)
    cys.append(cT9y)
    cxs.append(cT10x)
    cys.append(cT10y)
    cxs.append(cT11x)
    cys.append(cT11y)
    cxs.append(cT12x)
    cys.append(cT12y)
    cxs.append(cL1x)
    cys.append(cL1y)
    cxs.append(cL2x)
    cys.append(cL2y)
    cxs.append(cL3x)
    cys.append(cL3y)
    cxs.append(cL4x)
    cys.append(cL4y)
    cxs.append(cL5x)
    cys.append(cL5y)
    cxs.append(cS1x)
    cys.append(cS1y)
    cxs.append(cS1bx)
    cys.append(cS1by)

    
    spl = UnivariateSpline(cys,cxs)
    xs = np.linspace(cys[0], cys[-1], 100)
    spl.set_smoothing_factor(0)
    ysp = spl(xs).tolist()
    ##ysp = cxs
    ##xs = cys
    
    pts = list(map(list, zip(ysp, xs)))

    pts = np.array(pts,np.int32)
    ptspl = pts.reshape((-1, 1, 2))

    cv2.polylines(image,[ptspl], 0, (250,15,15), 1)  
    
    Circle(image,(cS1t[0],cS1t[1]),2,(10,180,230),-1)
    Circle(image,(cL5[0],cL5[1]),2,(200,200,20),-1)
    Circle(image,(cL4[0],cL4[1]),2,(200,200,20),-1)
    Circle(image,(cL3[0],cL3[1]),2,(200,200,20),-1)
    Circle(image,(cL2[0],cL2[1]),2,(200,200,20),-1)
    Circle(image,(cL1[0],cL1[1]),2,(200,200,20),-1)

    Circle(image,(cT12[0],cT12[1]),2,(50,50,180),-1)
    Circle(image,(cT11[0],cT11[1]),2,(50,50,180),-1)
    Circle(image,(cT10[0],cT10[1]),2,(50,50,180),-1)
    Circle(image,(cT9[0],cT9[1]),2,(50,50,180),-1)
    Circle(image,(cT8[0],cT8[1]),2,(50,50,180),-1)
    Circle(image,(cT7[0],cT7[1]),2,(50,50,180),-1)
    Circle(image,(cT6[0],cT6[1]),2,(50,50,180),-1)
    Circle(image,(cT5[0],cT5[1]),2,(50,50,180),-1)
    Circle(image,(cT4[0],cT4[1]),2,(50,50,180),-1)
    Circle(image,(cT3[0],cT3[1]),2,(50,50,180),-1)
    Circle(image,(cT2[0],cT2[1]),2,(50,50,180),-1)
    Circle(image,(cT1[0],cT1[1]),2,(50,50,180),-1)

    Circle(image,(cC7[0],cC7[1]),2,(30,180,10),-1)
    Circle(image,(cC6[0],cC6[1]),2,(30,180,10),-1)
    Circle(image,(cC5[0],cC5[1]),2,(30,180,10),-1)
    Circle(image,(cC4[0],cC4[1]),2,(30,180,10),-1)
    Circle(image,(cC3[0],cC3[1]),2,(30,180,10),-1)
    #Circle(image,(cC2[0],cC2[1]),2,(30,180,10),-1)


    pos = 0

    pt1 = [cC7x,2*height]
    Line(image,(cC7[0],cC7[1]),pt1,(255,255,255), 1)#prumo
    #================================================================
   
    #os.system('cls' if os.name == 'nt' else 'clear')
    print('----------------------------------------------------------------')
    print('Pelvis')
    print('----------------------------------------------------------------')
    ssAngle = 180-abs(180*Angle([vS1[2][0],vS1[2][1]],[vS1[3][0],vS1[3][1]])/np.pi)
   
    
    #----------------
    ptAngle = 180*Angle(cFemur,cS1t)/np.pi
    if(flagRight):
        if(ptAngle>0):
            ptAngle = ptAngle - 90
        else:
            ptAngle = ptAngle + 90
    else:
        if(ptAngle>0):
            ptAngle = 90 - ptAngle
        else:
            ptAngle = -90 - ptAngle
    print('Pelvic Tilt (PT): ',f"{abs(ptAngle):0.1f}")
    print('Pelvic Incidence (PI): ',f"{abs(ptAngle)+abs(ssAngle):0.1f}" )
    print('Sacral Slope (SS): ',f"{abs(ssAngle):0.1f}")
    #angleSS = AngleBetweenLines((vS1[0],vS1[1]),(vS1[0],cC7))*180/np.pi
    angleS = Angle(vS1[2],vS1[3])*180/np.pi
    angleSC7 = Angle(vS1[2],cC7)*180/np.pi
    print('Spino Sacral Angle = ',f"{abs(angleSC7-angleS):.1f}")

    Line(image,cFemur,cS1t,(255,240,5), 1)
    Line(image,vS1[2],vS1[3],(0,240,255), 1)
    
    cv2.putText(image,"SS:"+ ('% 0.1f' % abs(ssAngle)), (round(width/6),round(cFemury-15)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 240, 0), 1, cv2.LINE_AA)
    cv2.putText(image,"PT:"+ ('% 0.1f' % abs(ptAngle)), (round(width/6),round(cFemury-45)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 240, 0), 1, cv2.LINE_AA)
    cv2.putText(image,"PI:"+ ('% 0.1f' % (abs(ptAngle)+ abs(ssAngle))), (round(width/6),round(cFemury-30)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 240, 0), 1, cv2.LINE_AA)
    

    basename = os.path.basename(dirName)
    data = [basename,abs(ssAngle), abs(ptAngle), (abs(ptAngle)+abs(ssAngle)), abs(angleSC7-angleS)]
    #-------------------------------------------------------------------
       
    curv = Curvature(xs, [-y for y in ysp])
    #refDist = Distance(cT1,cS1)
    
    
    
    
    
    infls = np.where(np.diff(np.sign(curv)))[0]
   

    nx,ny = Normal(ysp, xs)
    nx2 = [x * 200 for x in nx]
    ny2 = [y * 200 for y in ny]
    ptsNorm = list(map(list, zip(nx2, ny2)))
    #print('norm',ptsNorm)
    ptsNorm = np.array(ptsNorm,np.int32)
    #ptsNorm = ptsNorm.reshape((-1, 1, 2))
    
    #cv2.polylines(image,[ptspl+ptsNorm], 0, (100,255,0), 1)
    #cv2.polylines(image,[ptspl-ptsNorm], 0, (100,255,0), 1)

      
    #colScl = refDist/30000
    cxsAux=[]
    cysAux=[]
    for i in range(len(cxs)-1):
        cxsAux.append(cxs[i]-cC7x)
        cysAux.append(metricRef*(cys[0]+cys[-1]-cys[i]))
    
    ###plt.rcParams.update({'font.size': 25})
    
    plt.plot([y*metricRef  for y in cxsAux],cysAux,'o',markersize=5)###20)
    
    a=[y*metricRef  for y in cxsAux]
    
    #plt.annotate("C2", (a[0],cysAux[0]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("C3", (a[0],cysAux[0]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("C4", (a[1],cysAux[1]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("C5", (a[2],cysAux[2]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("C6", (a[3],cysAux[3]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("C7", (a[4],cysAux[4]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T1", (a[5],cysAux[5]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T2", (a[6],cysAux[6]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T3", (a[7],cysAux[7]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T4", (a[8],cysAux[8]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T5", (a[9],cysAux[9]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T6", (a[10],cysAux[10]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T7", (a[11],cysAux[11]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T8", (a[12],cysAux[12]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T9", (a[13],cysAux[13]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T10", (a[14],cysAux[14]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T11", (a[15],cysAux[15]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("T12", (a[16],cysAux[16]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("L1", (a[17],cysAux[17]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("L2", (a[18],cysAux[18]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("L3", (a[19],cysAux[19]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("L4", (a[20],cysAux[20]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("L5", (a[21],cysAux[21]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    plt.annotate("S1", (a[22],cysAux[22]),textcoords="offset points",xytext=(-10,0),ha='center',fontsize=7)
    

    plt.plot([(y-cC7x)*metricRef  for y in ysp],(cys[0]+cys[-1]-xs)*metricRef,'-b',label="Spinal column",linewidth=2,markersize=12)
    
    plt.xlim(-metricRef*width/2,metricRef*width/2)
    plt.ylim(0,metricRef*height)
    
    #------------------------------------------------------------------------
    #Thoracic kyphosis (Funtional):
    #------------------------------------------------------------------------
    if(len(infls)<2):
        pt0 = (pts[0,0], pts[0,1])
    else:   
        infpts=[]
        for i in range(len(infls)):
            infpts.append([pts[infls[i],0], pts[infls[i],1]])    
        idx1,d =  MinDistance((cT1x,cT1y),infpts)    
        idx2,d =  MinDistance((cL1x,cL1y),infpts)
        pos = infls[idx1]
        pt0 = (pts[pos,0], pts[pos,1])
    pt1 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line1 = (pt0,pt1)
    Circle(image,pt0,3,(0,0,255),-1)
    
   
    pos = infls[idx2]
    pt2 = (pts[pos,0], pts[pos,1])
    pt3 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line2 = (pt2,pt3)
    Circle(image,pt2,3,(0,0,255),-1)
    ptix,ptiy = LineIntersection(line1,line2)
    angle = AngleBetweenLines(line1,line2)
    angle = 180 * angle/np.pi
    print('----------------------------------------------------------------')
    print('Dynamic spine shape (Inflection Point)')
    print('----------------------------------------------------------------')
    print("Thoracic kyphosis:\nCervical inflection point to lumbar inflection point: ", f"{abs(angle):.1f}")
    
    pti = np.array((ptix,ptiy),np.int32)
    pti = tuple(pti)

    Line(image,pt0,pti,(0,0,255), 1)
    Line(image,pt2,pti,(0,0,255), 1)
    
    ptixAux = ptix
    ptiyAux = ptiy
    cv2.putText(image,'% 0.1f' % abs(angle), (round(min(ptixAux,width)-40),round(ptiyAux-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 9, 255), 1, cv2.LINE_AA)  
    data.append(abs(angle) )
    #------------------------------------------------------------------------------------------------------------------------ 
    #------------------------------------------------------------------------
    #Lumbar lordosis (Funtional):
    #------------------------------------------------------------------------
    pos = infls[idx2]
    pt0 = (pts[pos,0], pts[pos,1])
    pt1 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line1 = (pt0,pt1)
    line2 = ((round(vS1[2][0]),round(vS1[2][1])),(round(vS1[3][0]),round(vS1[3][1])))
    ptix,ptiy = LineIntersection(line1,line2)
    angle = AngleBetweenLines(line1,line2)
    angle = 180 * angle/np.pi
    data.append(abs(angle)) 
    print("Lumbar lordosis:\nlumbar inflection point to S1 plateau:", f"{abs(angle):.1f}")
    pti = np.array((ptix,ptiy),np.int32)
    pti = tuple(pti)
    #Line(image,(pt0[0],pt0[1]),(pti[0],pti[1]),(255,255,255), 5)
    #Line(image,(cS1tx,cS1ty),(pti[0],pti[1]),(255,255,255), 5)
    #cv2.putText(image,'% 0.1f' % abs(angle), (round(max(ptix,0)+0.1*width),round(ptiy)), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (0, 0, 255), 1, cv2.LINE_AA) 
    #------------------------------------------------------------------------
    pos = infls[idx2]
    pt0 = (pts[pos,0], pts[pos,1])
    pt1 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line1 = (pt0,pt1)

    
    pos,d =  MinDistance((cS1tx,cS1ty),pts)
    pos=pos-1
    pt2 = (pts[pos,0], pts[pos,1])
    pt3 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line2 = (pt2,pt3)

    ptix,ptiy = LineIntersection(line1,line2)
    angle = AngleBetweenLines(line1,line2)
    angle = 180 * angle/np.pi
    data.append(abs(angle)) 
    print("Lumbar lordosis:\nlumbar inflection point to S1 normal: ", f"{abs(angle):.1f}")
    pti = np.array((ptix,ptiy),np.int32)
    pti = tuple(pti)
    Line(image,(pt0[0],pt0[1]),(pti[0],pti[1]),(0,0,255), 1)
    Line(image,(cS1tx,cS1ty),(pti[0],pti[1]),(0,0,255), 1)
    cv2.putText(image,'% 0.1f' % abs(angle), (round(max(ptix,0)+0.2*width),round(ptiy)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA) 
    #------------------------------------------------------------------------
    '''
    pos = infls[idx2]
    pt0 = (pts[pos,0], pts[pos,1])
    pt1 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line1 = (pt0,pt1)
    
    
    pos,d =  MinDistance((cL5x,cL5y),pts)
    pt2 = (pts[pos,0], pts[pos,1])
    pt3 = (pts[pos,0]+ptsNorm[pos,0], pts[pos,1]+ptsNorm[pos,1])
    line2 = (pt2,pt3)
    ###line2=(vS1[0],vS1[1])

    
    ptix2,ptiy = LineIntersection(line1,line2)
    angle = AngleBetweenLines(line1,line2)
    angle = 180 * angle/np.pi
    data.append(angle)
    print("Lumbar lordosis:\nL5 center (normal)  to lumbar inflection point: ", f"{abs(angle):.1f}")
    pti = np.array((ptix,ptiy),np.int32)
    pti = tuple(pti)

    Line(image,(pt0[0],pt0[1]),(pti[0],pti[1]),(255,0,255), 5)
    Line(image,(cL5x,cL5y),(pti[0],pti[1]),(255,0,255), 5)

    cv2.putText(image,'% 0.1f' % angle, (round(max(ptix,0)+0.1*width),round(ptiy+5)), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (255, 0, 255), 5, cv2.LINE_AA) 
    '''
    #------------------------------------------------------------------------
    print('----------------------------------------------------------------')
    print('Anatomic spine shape')
    print('----------------------------------------------------------------')
    
    angle = AngleBetweenLines((vT1[2],vT1[3]),(vT12[0],vT12[1]))*180/np.pi
    data.append(abs(angle))
    print("Thoracic kyphosis:\nT1 Upper Plateau - T12 Lower Plateau: ",f"{abs(angle):3.1f}")
    cv2.putText(image,'% 0.1f' % abs(angle), (round(min(ptixAux,width)-40),round(ptiyAux)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (55, 200, 255), 1, cv2.LINE_AA)


    angle = AngleBetweenLines((vT1[2],vT1[3]),(vL1[2],vL1[3]))*180/np.pi
    data.append(abs(angle))
    print("Thoracic kyphosis:\nT1 Upper Plateau - L1 Upper Plateau: ",f"{abs(angle):3.1f}")

    angle = AngleBetweenLines((vT4[2],vT4[3]),(vT12[0],vT12[1]))*180/np.pi
    data.append(abs(angle))
    print("Thoracic kyphosis:\nT4 Upper Plateau - T12 Lower Plateau: ",f"{abs(angle):3.1f}")   
    
    angle = AngleBetweenLines((vT4[2],vT4[3]),(vL1[2],vL1[3]))*180/np.pi   
    data.append(abs(angle))
    print("Thoracic kyphosis:\nT4 Upper Plateau - L1 Upper Plateau (TK): ",f"{abs(angle):3.1f}")
      
    
    angle = AngleBetweenLines((vL1[2],vL1[3]),(vS1[2],vS1[3]))*180/np.pi
    data.append(abs(angle))
    print("Lumbar lordosis:\nL1 Upper Plateau - S1 Plateau (LL):\t",f"{abs(angle):3.1f}")
    ################################################                                  ############
    cv2.putText(image,'% 0.1f' % abs(angle), (round(max(ptix,0)+0.2*width),round(ptiy+0.025*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (55, 200, 255), 1, cv2.LINE_AA)
    ################################################                                  ############
    #------------------------------------------------------------------------
    #"Anatomic Lines"
    #------------------------------------------------------------------------
    line1 =(vT1[2],vT1[3])
    line2 = (vT12[0],vT12[1])
    ptix,ptiy = LineIntersection(line1,line2)
    pti = np.array((ptix,ptiy),np.int32)
    pti = tuple(pti)
    Line(image,(vT1[3][0],vT1[3][1]),(pti[0],pti[1]),(55,255,255), 1)
    Line(image,(vT12[1][0],vT12[1][1]),(pti[0],pti[1]),(55,255,255), 1)
    
    line1 =(vS1[2],vS1[3])
    line2 = (vL1[2],vL1[3])
    ptix,ptiy = LineIntersection(line1,line2)
    pti = np.array((ptix,ptiy),np.int32)
    pti = tuple(pti)
    Line(image,(cS1tx,cS1ty),(pti[0],pti[1]),(55,255,255), 1)
    Line(image,(vL1[2][0],vL1[2][1]),(pti[0],pti[1]),(55,255,255), 1)
    #------------------------------------------------------------------------
    print('----------------------------------------------------------------')
    print('Balance')
    print('----------------------------------------------------------------')
    
    if(flagRight):
        C7PL = cC7x #plumb
        sacroAntx = vS1[3][0]
        SVA =  C7PL - sacroAntx
        SFD  = abs(cFemurx - sacroAntx)
        ratio = 1000000
        if(SFD!=0):
            ratio = SVA/SFD
        print('SVA/SFD ratio = ',f"{ratio:.2f}")
        print('SVA: Sagital Vertical Axis (Sacro Plumb Line distance)')
        print('SFD: Sacro Femoral distance')
        d = Distance((cFemur[0],1),(vS1[0][0],1))

        Line(image,(vS1[3][0]-d-0.1*width,cFemur[1]),(cFemur[0]+0.1*width,cFemur[1]),(255,255,255), 1)
        Line(image,(vS1[3][0],cFemur[1]-3),(vS1[3][0],cFemur[1]+3),(0,255,255), 1)
        cv2.putText(image,"BL:"+ ('% 0.2f' % ratio), (round(width/6),round(cFemury)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        
        
        #cv2.putText(image,'% 0.2f' % ratio, (int(round(C7PL+0.005*width)),int(0.994*height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, Line_AA) #ID 53
        
    
    data.append(ratio) 
   
    print('----------------------------------------------------------------')
    print('Individual Angles:')
    print('----------------------------------------------------------------')
    #angle = Angle(vC7[0],vC7[1])*180/np.pi
    #print("C7UpperPlateau:\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vC7[2],vC7[3]),(vC7[0],vC7[1]))*180/np.pi
    data.append(angle)
    print("C7 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vC7[0],vC7[1]),(vT1[2],vT1[3]))*180/np.pi
    data.append(angle)
    print("C7-T1 (disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT1[2],vT1[3]),(vT1[0],vT1[1]))*180/np.pi
    data.append(angle)
    print("T1 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT1[0],vT1[1]),(vT2[2],vT2[3]))*180/np.pi
    data.append(angle)
    print("T1-T2(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT2[2],vT2[3]),(vT2[0],vT2[1]))*180/np.pi
    data.append(angle)
    print("T2 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT2[0],vT2[1]),(vT3[2],vT3[3]))*180/np.pi
    data.append(angle)
    print("T2-T3(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT3[2],vT3[3]),(vT3[0],vT3[1]))*180/np.pi
    data.append(angle)
    print("T3 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT3[0],vT3[1]),(vT4[2],vT4[3]))*180/np.pi
    data.append(angle)
    print("T3-T4(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT4[2],vT4[3]),(vT4[0],vT4[1]))*180/np.pi
    data.append(angle)
    print("T4 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT4[0],vT4[1]),(vT5[2],vT5[3]))*180/np.pi
    data.append(angle)
    print("T4-T5(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT5[2],vT5[3]),(vT5[0],vT5[1]))*180/np.pi
    data.append(angle)
    print("T5 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT5[0],vT5[1]),(vT6[2],vT6[3]))*180/np.pi
    data.append(angle)
    print("T5-T6(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT6[2],vT6[3]),(vT6[0],vT6[1]))*180/np.pi
    data.append(angle)
    print("T6 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT6[0],vT6[1]),(vT7[2],vT7[3]))*180/np.pi
    data.append(angle)
    print("T6-T7(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT7[2],vT7[3]),(vT7[0],vT7[1]))*180/np.pi
    data.append(angle)
    print("T7 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT7[0],vT7[1]),(vT8[2],vT8[3]))*180/np.pi
    data.append(angle)
    print("T7-T8(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT8[2],vT8[3]),(vT8[0],vT8[1]))*180/np.pi
    data.append(angle)
    print("T8 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT8[0],vT8[1]),(vT9[2],vT9[3]))*180/np.pi
    data.append(angle)
    print("T8-T9(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT9[2],vT9[3]),(vT9[0],vT9[1]))*180/np.pi
    data.append(angle)
    print("T9 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT9[0],vT9[1]),(vT10[2],vT10[3]))*180/np.pi
    data.append(angle)
    print("T9-T10(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT10[2],vT10[3]),(vT10[0],vT10[1]))*180/np.pi
    data.append(angle)
    print("T10 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT10[0],vT10[1]),(vT11[2],vT11[3]))*180/np.pi
    data.append(angle)
    print("T10-T11(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT11[2],vT11[3]),(vT11[0],vT11[1]))*180/np.pi
    data.append(angle)
    print("T11 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT11[0],vT11[1]),(vT12[2],vT12[3]))*180/np.pi
    data.append(angle)
    print("T11-T12(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT12[2],vT12[3]),(vT12[0],vT12[1]))*180/np.pi
    data.append(angle)
    print("T12 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vT12[0],vT12[1]),(vL1[2],vL1[3]))*180/np.pi
    data.append(angle)
    print("T12-L1(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL1[2],vL1[3]),(vL1[0],vL1[1]))*180/np.pi
    data.append(angle)
    print("L1 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL1[0],vL1[1]),(vL2[2],vL2[3]))*180/np.pi
    data.append(angle)
    print("L1-L2(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL2[2],vL2[3]),(vL2[0],vL2[1]))*180/np.pi
    data.append(angle)
    print("L2 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL2[0],vL2[1]),(vL3[2],vL3[3]))*180/np.pi
    data.append(angle)
    print("L2-L3(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL3[2],vL3[3]),(vL3[0],vL3[1]))*180/np.pi
    data.append(angle)
    print("L3 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL3[0],vL3[1]),(vL4[2],vL4[3]))*180/np.pi
    data.append(angle)
    print("L3-L4(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL4[2],vL4[3]),(vL4[0],vL4[1]))*180/np.pi
    data.append(angle)
    print("L4 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL4[0],vL4[1]),(vL5[2],vL5[3]))*180/np.pi
    data.append(angle)
    print("L4-L5(disc):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL5[2],vL5[3]),(vL5[0],vL5[1]))*180/np.pi
    data.append(angle)
    print("L5 (vertebra):\t\t",f"{angle:+3.1f}")
    angle = AngleBetweenLines((vL5[0],vL5[1]),(vS1[2],vS1[3]))*180/np.pi
    data.append(angle)
    print("L5-S1(disc):\t\t",f"{angle:+3.1f}")
    
    #-------------------------------------------------------------
    #print(os.name)
    
    
    with open('SpineData.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        
        if os.stat("SpineData.csv").st_size == 0:
            
            header = ["ID","Sacral Slope", "Pelvic Tilt", "Pelvic Incidence", "Spino Sacral Angle", "Dynamic kyphosis", "Dynamic lordosis_S1_Plateau ", "Dynamic lordosis ", "Anat.kyphosis T1-T12", "Anat.kyphosis T1-L1", "Anat.kyphosis T4-T12", "Anat.kyphosis T4-L1","Anatomic lordosis", "SVA/SFD ratio","VC7","C7-T1","VT1","T1-T2","VT2","T2-T3","VT3","T3-T4","VT4","T4-T5","VT5","T5-T6","VT6","T6-T7","VT7","T7-T8","VT8","T8-T9","VT9","T9-T10","VT10","T10-T11","VT11","T11-T12","VT12","T12-L1","VL1","L1-L2","VL2","L2-L3","VL3","L3-L4","VL4","L4-L5","VL5","L5-S1"]
            writer.writerow(header)
        
        writer.writerow(data)
     
    if(os.name=='nt'or os.name=='posix'):
        resizedImg =  cv2.resize(image,(int(width/scl),(int(height/scl))))
        cv2.imshow('Zoom', resizedImg)

    cv2.imshow('Image', image)
    
    filename = dirName+'/'+fname + '_res.jpg'
    cv2.imwrite(filename, image)
    

    #
    #------------------
    h=height
    m=metricRef
    plt.plot(curv*height*metricRef*10,(cys[0]+cys[-1]-xs)*metricRef,'-',linewidth=2,label="Curvature (x10)")
    plt.plot((ysp[infls[idx1]] -cC7x)*metricRef,(cys[0]+cys[-1]-xs[infls[idx1]])*metricRef,'or',label="Inflection Point",markersize=5)
    plt.plot((ysp[infls[idx2]] -cC7x)*metricRef,(cys[0]+cys[-1]-xs[infls[idx2]])*metricRef,'or',markersize=5)
    if(flagRef):
        plt.xlabel("Distance (mm)")
        plt.ylabel("Distance (mm)")
    else:
        plt.xlabel("Distance (pixel)")
        plt.ylabel("Distance (pixel)")
    plt.legend(loc="lower left",fontsize=7)
    
    '''
   
    
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
    
    plt.rcParams.update({'axes.labelsize': 25})

    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    '''

    #plt.tight_layout()
    plt.axis('equal')
    plt.grid()
    #plt.show()
    filename = dirName+'/'+fname + '_grafico.svg'
    plt.savefig(filename)
    plt.close()
    key=cv2.waitKey(0)
     

#------------------------------------------------------------------
if __name__ == "__main__":
    main()

