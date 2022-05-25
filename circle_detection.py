import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_radius(centre, filtered_points):
    """
    Calcualte Radius given points on the circumference and the centre of the circle.

    centre : tuple
        Coordinates(index) of the centre of the circle in the image.
    
    filtered_points : array
        Points on circumference of circle with outliers removed.
    """
    avg_dist = 0
    avg_dist = 0
    for p in filtered_points:
        distance = np.sqrt((centre[0] - p[0])**2 + (centre[1] - p[1])**2)
        avg_dist += distance
    avg_dist/=len(filtered_points)
    return int(avg_dist)

def find_centre(filtered_points):
    """
    Calcualte centre of the circle in the image using points on circumference.

    filtered_points : array
        Points on circumference of circle with outliers removed.
    """
    centre = [0, 0]
    for p in filtered_points:
        centre[0] += p[0]
        centre[1] += p[1]
    centre[0] /= len(filtered_points)
    centre[1] /= len(filtered_points)
    centre = [int(c) for c in centre]
    
    return tuple(centre)

def filter_points(img, points, outlier_ratio = 30):
    """
    Provided image and detected points on circumerence on the circle, detects outliers in the circle points
    and replaces them with more reasonable points, ie. the point in the middle of the centre and the outlier.

    img : ndarray
        The image read in by cv2 in for of ndarray.
    
    points: array
        Array of original detected points.

    outlier_ratio : int
        Outlier ratio. Increasing it causes outlier detecting threshold to increase, causing more points 
        to be classified as outliers.
    """
    filtered_points = []
    height = img.shape[0]
    width = img.shape[1]
    h = height / outlier_ratio
    w = width / outlier_ratio
    for p in points:
        if (h <= p[1] <= (height-h)) and (w <= p[0] <= (width-w)) :
            filtered_points.append(p)
        else:
            filtered_points.append(((p[0] + (width / 2)) / 2, (p[1] + (height / 2)) / 2))
            print("Filtering")
    return filtered_points

def extract_points(img, px_intensity_thresh = 100):
    """
    Finds and extracts 8 points in the circle circumference based on changing pixel values.
    Assumes the outside of circle generally has lower pixel values(closer to 0 or black)
    than inside of the circle

    img : ndarray
        The image read in by cv2 in for of ndarray.

    px_intensity_thresh: int
        Pixel intensity threshold value for detecting points on circumference based on brightness.
        Increasing this value causes circle to expand towards more darker areas.
    """
    grey_thresh = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    px = grey_thresh.shape[0] * grey_thresh.shape[1]
    avg_px_intensity = int(np.sum(grey_thresh)/px)

    thresh_val = avg_px_intensity**2/px_intensity_thresh
    gre_thresh_bin = np.float32(grey_thresh > thresh_val)
    
    grey_thresh_bin_eroded = cv2.dilate(np.float32(grey_thresh > thresh_val), np.ones((15, 15), np.uint8))

    mid_row_ind = int(grey_thresh_bin_eroded.shape[0] / 2)
    mid_col_ind = int(grey_thresh_bin_eroded.shape[1] / 2)

    mid_row = grey_thresh_bin_eroded[mid_row_ind, :]
    mid_col = grey_thresh_bin_eroded[:, mid_col_ind]
    p1 = (np.where(mid_row == 1)[0][0], mid_row_ind)
    p2 = (np.where(mid_row == 1)[0][-1], mid_row_ind)

    p3 = (mid_col_ind, np.where(mid_col == 1)[0][0])
    p4 = (mid_col_ind, np.where(mid_col == 1)[0][-1])

    row_col_rat = img.shape[0]/img.shape[1]

    left_diag_idxs = list(zip(list(map(lambda x: round(x * row_col_rat), list(range(img.shape[1])))), 
                            list(range(img.shape[1]))))
    left_diag = np.array([grey_thresh_bin_eroded[r, c] for r, c in left_diag_idxs])

    p5 = left_diag_idxs[np.where(left_diag == 1)[0][0]]
    p6 = left_diag_idxs[np.where(left_diag == 1)[0][-1]]
    p5 = (p5[1], p5[0])
    p6 = (p6[1], p6[0])

    right_diag_idxs = list(zip(list(map(lambda x: int(x * row_col_rat), list(range(img.shape[0])))), 
                            list(range(img.shape[1] - 1, -1, -1))))
    right_diag = np.array([grey_thresh_bin_eroded[r, c] for r, c in right_diag_idxs])


    p7 = right_diag_idxs[np.where(right_diag == 1)[0][0]]
    p8 = right_diag_idxs[np.where(right_diag == 1)[0][-1]]
    p7 = (p7[1], p7[0])
    p8 = (p8[1], p8[0])
    return [p1, p2, p3, p4, p5, p6, p7, p8]

def detect_fish_circle(img):
    """
    Assumes there is a circle with lighter pixels inside and darker pixels outside.
    Detects the circle based on 8 points on the circumference.
    Calculates the centre and the radius of the circle.
    Writes the resultant image on disk.
    """

    #Find the circle in image and extract circumference points
    points = extract_points(img, px_intensity_thresh = 120)

    #Draw at all points detected on circumference
    # for p in points:
    #     img = cv2.circle(img, p, 20, (255, 0, 0), -1)

    #replace outliers with more reasonable values
    filtered_points = filter_points(img, points, outlier_ratio = 30)

    #find centre based on filtered points
    centre = find_centre(filtered_points)

    #calculate radius using filtered points
    radius = calculate_radius(centre, filtered_points)

    return centre, radius
    
