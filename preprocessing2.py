import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

image_path = "fox.png"

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

#m=cv.medianBlur(gray_image,3)
#m = median(gray_image)
#g = gaussian(gray_image)
#save(m,"median5.png")
#display2(g,"median")

#Histogram
#hist = cv.equalizeHist(m)
#display2(hist,"hist")

#Binarization/Thresholding

#f, otsu = cv.threshold(gray_image,70,255,cv.THRESH_OTSU)
#thresh, binary = cv.threshold(m,110,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C) #110 good
binary = cv.adaptiveThreshold(gray_image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,17,6.5) #110 good

display2(binary,"aa1a")
#display2(binary,"Gauss")
#save(binary,"lightning3_mean.png")

#display2(binary,"binary")

#invert
inverted = cv.bitwise_not(binary)
#display2(inverted,"inverted")
#save(inverted,"lighting3_gauss.png")

#display2(binary)

def dilate(image, k):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(k,k))
    img = cv.dilate(image,kernel,iterations=1)
    return img

def erode(image, k):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(k,k))
    img = cv.erode(image,kernel,iterations=1)
    return img

#Erode + dilate
def morph(image, k):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(k,k))
    img = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    return img


erosion_kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
erosion_kernel2 = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9))

morphed = cv.erode(inverted,erosion_kernel,iterations=1)
morphed = cv.dilate(morphed,dilate_kernel,iterations=1)


#eroded = erode(inverted,3)
#dilated = dilate(eroded,9)
#display2(dilated,"morph")
#morphed = morph(binary,5)
#display2(morphed,"morphed")

display2(morphed,"tmp")


def removeBorder(image):
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cntSorted = sorted(contours,key=lambda x:cv.contourArea(x))
    cnt = cntSorted[-1]
    x,y,w,h = cv.boundingRect(cnt)
    crop = image[y:y+h,x:x+w]
    return crop

crop = removeBorder(morphed)
#display2(crop,"crop")

contours, hierarchy = cv.findContours(crop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#save(crop,"mk2segmentation.png")

iter=0
for cont in contours:
    x,y,w,h = cv.boundingRect(cont)
    rect = cv.rectangle(crop,(x,y),(x+w,y+h),(255,255,255),5)
    #crop = dilated[y:y+h,x:x+w]
    print(iter)
    iter=iter+1

display2(crop,"rectangles")





