

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from myFunctions import *

def interpolate(flag):
    global vertices,imgCrop,imgCrop0,roi, scl, cropScl,reImWc
    x1=[]
    y1=[]
    for p in vertices:
        x1.append(p[1])
        y1.append(p[0])
    
    idx =np.argsort(x1)
    x1.remove(x1[idx[len(idx)-1]])
    y1.remove(y1[idx[len(idx)-1]])
    idx =np.argsort(x1)
    x1.remove(x1[idx[len(idx)-1]])
    y1.remove(y1[idx[len(idx)-1]])
    idx =np.argsort(x1)
    x1.remove(x1[idx[len(idx)-1]])
    
    y1.remove(y1[idx[len(idx)-1]])
    idx =np.argsort(x1)
    x1.remove(x1[idx[len(idx)-1]])
    y1.remove(y1[idx[len(idx)-1]]) 
    
    idx2 =np.argsort(x1)
    
    x=[]
    y=[]
    for i in range(len(idx2)-2):
        p = idx2[i]
        x.append(x1[p])
        y.append(y1[p])

    xnp=np.array(x)
    ynp=np.array(y)
    xn = xnp[:, np.newaxis]
    yn = ynp[:, np.newaxis]
    polynomial_features= PolynomialFeatures(degree=5)
    x_poly = polynomial_features.fit_transform(xn)

    model = LinearRegression()
    model.fit(x_poly, yn)
    yp = model.predict(x_poly)
    for i in range(len(xn)-1):
        v0x = float((xn[i]/scl))
        v0y = float((yp[i]/scl))
        v1x = float((xn[i+1]/scl))
        v1y = float((yp[i+1]/scl))
        Line(reImWc,[v0y,v0x],[v1y,v1x],(0,0,255), 1)
    
    xa=[]
    ya=[]
    xb=[]
    yb=[]
    for i in range(len(x)):
        if y[i]>yp[i]:
            xa.append(x[i])
            ya.append(y[i])
        else:    
            xb.append(x[i])
            yb.append(y[i])



    tck,u = splprep([xa,ya],k=2,s=5)
    u=np.linspace(0,1,num=150,endpoint=True)
    outa = splev(u,tck)

    tck,u = splprep([xb,yb],k=2,s=5)
    u=np.linspace(0,1,num=150,endpoint=True)
    outb = splev(u,tck)

    #mx = 1.1*max(x)   
    #x = [mx-v for v in x]
    #xa = [mx-v for v in xa]
    #outa[0] = [mx-v for v in outa[0]]
    #outb[0] = [mx-v for v in outb[0]]
    #plt.plot(y,x, 'ro',ya,xa,'go',outa[1],outa[0],'g',outb[1],outb[0],'r')
    #plt.show()
    if flag==0:
        for i in range(len(outa[0])-2):
            p0x = outa[0][i]/scl
            p0y = outa[1][i]/scl
            p1x = outa[0][i+1]/scl
            p1y = outa[1][i+1]/scl
            Line(reImWc,[p0y,p0x],[p1y,p1x],(0,240,255), 1)
        for i in range(len(outb[0])-2):
            p0x = outb[0][i]/scl
            p0y = outb[1][i]/scl
            p1x = outb[0][i+1]/scl
            p1y = outb[1][i+1]/scl
            Line(reImWc,[p0y,p0x],[p1y,p1x],(0,240,255), 1)    
    else:
        
        for i in range(len(outa[0])-2):
            p0x = cropScl*(outa[0][i]/scl-roi[1])
            p0y = cropScl*(outa[1][i]/scl-roi[0])
            p1x = cropScl*(outa[0][i+1]/scl-roi[1])
            p1y = cropScl*(outa[1][i+1]/scl-roi[0])
            Line(imgCrop,[p0y,p0x],[p1y,p1x],(0,240,255), 1)
        for i in range(len(outb[0])-2):
            p0x = cropScl*(outb[0][i]/scl-roi[1])
            p0y = cropScl*(outb[1][i]/scl-roi[0])
            p1x = cropScl*(outb[0][i+1]/scl-roi[1])
            p1y = cropScl*(outb[1][i+1]/scl-roi[0])
            Line(imgCrop,[p0y,p0x],[p1y,p1x],(0,240,255), 1)        
#####################

#------------------------------------------------------------------
# mouse event response
def click_event(event, x, y, flags, params):
    global vertices,imgCrop,imgCrop0,roi, scl, cropScl
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append([scl*(x/cropScl+roi[0]),scl*(y/cropScl+roi[1])])
        cv2.circle(imgCrop,(x,y),round(2*cropScl),(50,50,255),-11)
    if event == cv2.EVENT_RBUTTONDOWN:        
        imgCrop = imgCrop0.copy()
        for v in vertices:
            if((x/cropScl+roi[0])*scl>v[0]-4 and (x/cropScl+roi[0])*scl<v[0]+4 and (y/cropScl+roi[1])*scl>v[1]-4 and (y/cropScl+roi[1])*scl<v[1]+4):
                vertices.remove(v)
        for v in vertices:
            cv2.circle(imgCrop,(round(cropScl*(v[0]/scl-roi[0])),round(cropScl*(v[1]/scl-roi[1]))),round(2*cropScl),(50,50,255),-1)
    #if len(vertices)>=78:
    #    interpolate(1)    
    cv2.imshow('Image', imgCrop)
#------------------------------------------------------------------
def main():
    global vertices,imgCrop,imgCrop0,roi,scl,cropScl,reImWc
    screen = get_monitors()[0]
    Tk().withdraw()
    file = askopenfilename(initialfile = "*.*", title = "Select Dicon or Image file", filetypes = (("image File",(".dcm",".png")),("all files",".*")))
    dirname = os.path.dirname(file)
    fileName = os.path.basename(file)
    flagDcm = False
    fname, fextension = os.path.splitext(fileName)


    if fextension.lower() == ".dcm" or fextension == "":
        flagDcm = True
    if flagDcm:
        dcm = dcmread(file)
        image = np.uint8(dcm.pixel_array*0.002)
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height,width, channels = image.shape
    else:
        image = cv2.imread(file)
        height,width,channels = image.shape
    flagRef = False
    cntRef = 0
    scl = 1.1*max(width/screen.width,height/screen.height)    
    try:
        open_file = open(dirname+'/'+fileName + '_vertices.pkl', "rb")
        vertices = pickle.load(open_file)
        open_file.close()
    except:
        vertices=[]
        
    resizedImg =  cv2.resize(image,(round(width/scl),(round(height/scl))))
    reImWc = resizedImg.copy()
    for center in vertices:
         cv2.circle(reImWc,(round(center[0]/scl),round(center[1]/scl)),3,(50,50,255),-1)
    cropScl = 3
    resizedImg2Crop =  cv2.resize(image,(round(cropScl*width/scl),(round(cropScl*height/scl))))

   

    key = 0
    while(key != 27 ):
        #if len(vertices)>=78:
        #    interpolate(0)
        roi = cv2.selectROI(reImWc)
        imgCrop0 = resizedImg2Crop[round(cropScl*roi[1]):round(cropScl*(roi[1]+roi[3])), round(cropScl*roi[0]):round(cropScl*(roi[0]+roi[2]))]
        imgCrop0 = cv2.cvtColor(imgCrop0, cv2.COLOR_RGB2GRAY)
        imgCrop0 = cv2.equalizeHist(imgCrop0)
        imgCrop0 = cv2.cvtColor(imgCrop0, cv2.COLOR_GRAY2RGB)
        imgCrop=imgCrop0.copy()
        for v in vertices:
            cv2.circle(imgCrop,(round(cropScl*(v[0]/scl-roi[0])),round(cropScl*(v[1]/scl-roi[1]))),round(2*cropScl),(50,50,200),-1)

        #if len(vertices)>=78:
        #    interpolate(1)

        cv2.imshow('Image', imgCrop)
        cv2.setMouseCallback('Image', click_event)
        key=cv2.waitKey(0)
        cv2.destroyWindow('Image') 
        reImWc = resizedImg.copy()
        for center in vertices:
            cv2.circle(reImWc,(round(center[0]/scl),round(center[1]/scl)),3,(50,50,255),-1)
        open_file = open(dirname+'/'+fileName + '_vertices.pkl', "wb")
        pickle.dump(vertices, open_file)
        open_file.close()
#------------------------------------------------------------------
if __name__ == "__main__":
    main()
