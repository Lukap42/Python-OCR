import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

image_path = "lighting3.png"

def save(image, name):
    cv.imwrite("temp/"+name,image)

def display(image_path):
    dpi = 80
    im_data = plt.imread(image_path)
    height, width, depth = im_data.shape

    #What size does the figure need to be
    figsize = width/float(dpi), height/float(dpi)

    #Create figure with one axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])

    #Hide markings
    ax.axis('off')

    #display
    ax.imshow(im_data)

    plt.show()

def display2(image, name):
    dpi = 80
    height, width = image.shape

    #What size does the figure need to be
    figsize = width/float(dpi), height/float(dpi)

    #Create figure with one axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0,0,1,1])

    #Hide markings
    ax.axis('off')

    #display
    ax.imshow(image,cmap='gray')
    fig.canvas.manager.set_window_title(name)
    plt.show()


#Reading
img = cv.imread(image_path, cv.IMREAD_COLOR_RGB)

#Greyscale

gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#display2(gray_image,"gray")

def median(image):
    return cv.medianBlur(image,5)

def gaussian(image):
    return cv.GaussianBlur(image,(3,3),0)

m=cv.medianBlur(gray_image,5)
#m = median(gray_image)
#g = gaussian(gray_image)
#save(m,"median5.png")
#display2(g,"median")

#Histogram
hist = cv.equalizeHist(m)
#display2(hist,"hist")

#Binarization/Thresholding

thresh, binary = cv.threshold(m,110,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C) #110 good
#display2(binary,"Gauss")
#save(binary,"lightning3_mean.png")

#display2(binary,"binary")

#invert
inverted = cv.bitwise_not(binary)
#display2(inverted,"inverted")
#save(inverted,"lighting3_gauss.png")

#display2(binary)

def dilate(image, k):
    kernel = np.ones((k,k),np.uint8)
    img = cv.dilate(image,kernel,iterations=1)
    return img

def erode(image, k):
    kernel = np.ones((k,k),np.uint8)
    img = cv.erode(image,kernel,iterations=1)
    return img

#Erode + dilate
def morph(image, k):
    kernel = np.ones((k,k),np.uint8)
    img = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return img

#eroded = erode(binary,5)
#dilated = dilate(eroded,5)
#display2(dilated,"morph")
#morphed = morph(binary,4)
#display2(morphed,"morphed")


binary = cv.bitwise_not(binary)

display2(binary,"tmp")


def removeBorder(image):
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours,key=lambda x:cv.contourArea(x))
    cnt = cntSorted[-1]
    x,y,w,h = cv.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    return crop

crop = removeBorder(inverted)
#display2(crop,"crop")
crop = cv.bitwise_not(crop)

contours, hierarchy = cv.findContours(crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#save(crop,"mk2segmentation.png")

iter=0
for cont in contours:
    x,y,w,h = cv.boundingRect(cont)
    rect = cv.rectangle(crop,(x,y),(x+w,y+h),(255,255,255),5)
    if(h>180 or w>180):
        iter=iter-1
    #crop = dilated[y:y+h,x:x+w]
    print(iter)
    iter=iter+1

display2(crop,"rectangles")





