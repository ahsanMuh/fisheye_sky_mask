import os
from tqdm import tqdm
import numpy as np

from PIL import Image

from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.restoration import rolling_ball
from skimage.morphology import binary_opening, remove_small_objects
from skimage.util import invert
from skimage.measure import label, regionprops

MIN_SKY_SIZE = 0.18 # min percentage of sky size

def get_k_smallest():
    """ Returns the index of k smallest in the distance"""
    return np.where(distances == np.partition(distances, k_smallest)[k_smallest-1])[0][0]


def get_eucl_dis(point1, point2):
    ''' To get euclidean distance in x,y coord using np'''
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def get_min_dis(cont, centre):
    """
    Get the minimum distance of pixels in a contour with the
    centre of the image
    """
    pixel_x , pixel_y =np.where(cont.image)
    pixel_x += cont.bbox[0]
    pixel_y += cont.bbox[1]
    points = list(zip(pixel_x, pixel_y))
    dist = [get_eucl_dis(centre, loc) for loc in points]
    return min(dist)

def img_to_maskpath(org):

    # using background removal to get buildings (not clear enough)
    img = np.array(org)
    img = rgb2gray(img)
    print('converted the image into required format ...')
    background = rolling_ball(img)
    img = img - background
    print('Got the background ...')

    #image processing
    bn_open = binary_opening(img)
    bn_open_inv = invert(bn_open)
    bn_open_inv_big = remove_small_objects(bn_open_inv, min_size=1000)
    print('removed small objects ...')

    # getting contours
    bn_open_inv_big_lab = label(bn_open_inv_big)
    res = regionprops(bn_open_inv_big_lab)
    print('got the contours ...')

    # find the middle point
    centre = (img.shape[0]/2, img.shape[1]/2)
    distances = [get_min_dis(cont, centre) for cont in res]
    

    # get the right index
    for i in range(len(distances) - 1):
        ind = np.where(distances == np.partition(distances, i+1)[i])[0][0]
        area = res[ind].image.shape[0] * res[ind].image.shape[1]
        area_percent = area / (img.shape[0] * img.shape[1])
        if area_percent > MIN_SKY_SIZE:
            break

    print('Found the required contor ...')
    # now creating mask
    mask = np.zeros(img.shape)
    bbox = res[ind].bbox
    mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = res[ind].image
    print('about to save results ...')

    # in a landscape image, rotate the mask
    if org.width > org.height:
        mask = np.rot90(mask, 3)
    imsave('mask.png', mask)

    return 'mask.png'