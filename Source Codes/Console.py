import cv2
import os
import sys
import numpy as np
from numpy.lib.type_check import imag
import skfuzzy as fuzz
from skfuzzy import image

# main function of system
def FCM_segmentation(img_url):
    # create directory for outputs
    img_name = img_url.split('\\')[-1]
    img_name = img_name.split('.')[0]

    if not os.path.exists(img_name):
        os.mkdir(img_name)

    # image acqustiotion
    image = cv2.imread(img_url)
    image = cv2.resize(image,(256,256),cv2.INTER_AREA)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #plt.imsave('1ORGINAL_IMAGE.jpg',image,cmap='gray')

    #save orginal image
    cv2.imwrite(str(img_name)+'/Input.jpg',image)

    # iamge preprocessing
    # AMF
    img_median_sigmoid = AdaptiveMedianFilter(image)
    img_median = cv2.medianBlur(image, 5)
    cv2.imwrite('AMF.jpg', img_median_sigmoid)
    #plt.imsave('2IMAGE_MEDIAN.jpg',img_median_sigmoid,cmap='gray')

    # BCET
    image_bcet = BalanceContrastEnhancementTechnique(img_median)
    cv2.imwrite('img_bcet.jpg', image_bcet)
    #plt.imsave('3IMAGE_BCET.jpg',image_bcet,cmap='gray')

    # segmentation of tumor with thresholding
    thresh_f = Thresholding(image_bcet)
    #plt.imsave('4THRESH_F.jpg',thresh_f,cmap='gray')

    # segmentation of brain with fuzzy c-means
    image_FCM = FCM('img_bcet.jpg')
    #plt.imsave('5IMAGE_FCM.jpg',image_FCM,cmap='gray')

    # Combine results
    seg_img=segmentation(image_FCM,thresh_f)
    #plt.imsave('7SEGIMG.jpg',seg_img,cmap='gray')
    edges=dataanalysis(seg_img,img_name)
    segimg2 = cv2.addWeighted(edges, 0.5, image, 0.7, 0)
    #plt.imsave('6SEGIMG.jpg',edges,cmap='gray')
    cv2.imwrite(str(img_name)+'/Output.jpg',segimg2)

    # removing addition files
    os.remove('AMF.jpg')
    os.remove('border.jpg')
    os.remove('FCM.jpg')
    os.remove('img_bcet.jpg')

    return img_name
    


# AMF implementation
def AdaptiveMedianFilter(grayimage):
    try:
        img_out = grayimage.copy()
        height = grayimage.shape[0]
        width = grayimage.shape[1]
        for i in np.arange(6, height - 5):
            for j in np.arange(6, width - 5):
                neighbors = []
        
                for k in np.arange(-6, 6):
                    for l in np.arange(-6, 6):
                        a = grayimage.item(i + k, j + l)
                        neighbors.append(a)
        neighbors.sort()
        median = neighbors[30]
        b = median
        img_out.itemset((i, j), b)

    except Exception as e:
        print("Error=" + e.args[0])
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

    return img_out.astype(np.uint8)

def im2double(im):
    ''' Convert image to double precision '''
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

# BCET implementation
def BalanceContrastEnhancementTechnique(gray_image):
    x = im2double(gray_image) # INPUT IMAGE
    Lmin = np.min(x.ravel()) # MINIMUM OF INPUT IMAGE
    Lmax =np.max(x.ravel()) # MAXIMUM OF INPUT IMAGE
    Lmean = np.mean(x) # MEAN OF INPUT IMAGE
    LMssum = np.mean(pow(x,2)) # MEAN SQUARE SUM OF INPUT IMAGE
    
    Gmin = 0 # MINIMUM OF OUTPUT IMAGE
    Gmax = 255 # MAXIMUM OF OUTPUT IMAGE
    Gmean =85 # MEAN OF OUTPUT IMAGE 80 (Recomended)
    
    bnum = pow(Lmax,2) * (Gmean - Gmin) - LMssum * (Gmax - Gmin) + pow(Lmin,2) * (Gmax - Gmean)
    bden = 2 * (Lmax * (Gmean - Gmin) - Lmean * (Gmax - Gmin) + Lmin * (Gmax - Gmean))
    
    b = bnum / bden
    a = (Gmax - Gmin) / ((Lmax - Lmin) * (Lmax + Lmin - 2 * b))
    c = Gmin - a * pow((Lmin - b), 2)
    y = a *pow((x - b),2)+ c # PARABOLIC FUNCTION
    y = y.astype(np.uint8)
    
    return y

def change_color_fuzzycmeans(cluster_membership, clusters):
    '''set a color to each cluster in fcm function'''
    img = []
    for pix in cluster_membership.T:
        img.append(clusters[np.argmax(pix)])
    return img

def imfill(im_th):
    '''This function uses and updates the mask'''
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out

# DCM implementation
def FCM(image_bcet):
    list_img = []
    img =cv2.imread(str(image_bcet))
    
    rgb_img = img.reshape((img.shape[0] * img.shape[1], 3))
    list_img.append(rgb_img)
    n_data = len(list_img)
    clusters = [2]

    for index, rgb_img in enumerate(list_img):
        img = np.reshape(rgb_img, (256, 256, 3)).astype(np.uint8)
        shape = np.shape(img)
        # looping every cluster
        for i, cluster in enumerate(clusters):
            # Fuzzy C Means
            #new_time = time()
            
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(rgb_img.T, cluster, 2, error=0.005, maxiter=1000, init=None, seed=42)
            new_img = change_color_fuzzycmeans(u, cntr)
            fuzzy_img = np.reshape(new_img, shape).astype(np.uint8)
            ret, seg_img = cv2.threshold(fuzzy_img, np.max(fuzzy_img) - 1, 255,cv2.THRESH_BINARY)
            
            seg_img_1d = seg_img[:, :, 1]
            bwfim1 = bwareaopen(seg_img_1d, 500)
            bwfim2 = imclearborder(bwfim1)

            cv2.imwrite('border.jpg', bwfim2)
                

            bwfim3 = imfill(bwfim2)    

            cv2.imwrite('FCM.jpg', bwfim3)
    return bwfim3

def bwareaopen(imgBW, areaPixels):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # For each contour, determine its total occupying area
    for idx in np.arange(len(contours)):
        area = cv2.contourArea(contours[idx])
        if (area >= 0 and area <= areaPixels):
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)
        
    return imgBWcopy

def imclearborder(imgBW):
    # Given a black and white image, first find all of its contours
    radius = 2
    imgBWcopy = imgBW.copy()

    contours, hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]
    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]
        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]
            # If this is within the radius of the border
            # this contour goes bye bye!
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows - 1 - radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols - 1 - radius and colCnt < imgCols) 

            if check1 or check2:
                contourList.append(idx)
                break
        
        for idx in contourList:
            cv2.drawContours(imgBWcopy, contours, idx, (0, 0, 0), -1)           
    return imgBWcopy

# Thresholding fuction
def Thresholding(image_bcet):
    blur = cv2.GaussianBlur(image_bcet,(5,5),0)
    T, thresh_f = cv2.threshold(blur,160,200, cv2.THRESH_BINARY)
    return thresh_f

# add FCM image and thresholding image as final stage
def segmentation(fcm_image,ths_image):
    brain = fcm_image
    tumor = ths_image
    segimg = cv2.addWeighted(brain, 0.5, tumor, 0.7, 0)
    return segimg

# add color map to segmented image
def dataanalysis(seg_img,img_name):
    try:
        detected_edges = cv2.Canny(seg_img, 10, 10 * 3, 5)
        colour = cv2.applyColorMap(seg_img, cv2.COLORMAP_JET)
        cv2.imwrite(str(img_name)+'/Color.jpg', colour)
        return detected_edges
    except Exception as e:
        print("Error=" + e.args[0])
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)


n = len(sys.argv)

if (n <= 1):
    print('\n\n\n--------------------------------------------------\n\n')
    print('please enter filepath after program filename\n\n')
    print('Example: python Console.py ./Dataset/yes/Y1.jpg')
    print('\n\n--------------------------------------------------\n\n\n')
    exit(0)



img_url = sys.argv[1]

img_name = FCM_segmentation(img_url=img_url)

print('\n\nOutputs saved in {} folder in the current directory!\n\n'.format(img_name))