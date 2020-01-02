import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk


def file_browse():
    """
    Function to read in the files
    ----------------------------------------
    root.filename : Desired image file path
    """

    root = Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(parent=root, title="Select the Image")

    return root.filename


def get_filteredimage(image, lower, upper):
    """
    Function to calculate a mask which threshold the desired area
    ------------------------------------------------------------------
    image : Original Image

    lower : Lower threshold Limit

    upper : upper threshold limit
    ------------------------------------------------------------------
    rgb_filtered : Image after threshold
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    res, mask = cv2.threshold(gray, lower, upper, cv2.THRESH_BINARY)
    mask1 = cv2.inRange(image, (lower, lower, lower), (255, 255, 255))

    rgb_filtered = cv2.bitwise_and(image, image, mask=mask)

    return rgb_filtered


def plot_image(image, title="Plot"):
    """
    Function that plots the image
    -------------------------------------
    image : Image to be display

    title : title of the image
    """

    # cv2.imshow(title, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(image)
    plt.show()


def template_matching(image, temp, threshold):
    """
    Function tries matching the given template in the given image
    according to a threshold
    -------------------------------------------------------------
    image : Original image

    temp : Template Image

    threshold : The threshold at which the template is to be matched
    -------------------------------------------------------------
    _loc : Tuple of detected points
    """

    filt_block_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(filt_block_gray, temp, cv2.TM_CCOEFF_NORMED)
    _loc = np.where(res >= threshold)

    return _loc


def feature_matching(image, temp):
    """
    Function tries to match the features of the given template and
    the given image to find the desired object
    -------------------------------------------------------------
    image : Original image

    temp : Template Image
    -------------------------------------------------------------
    _matches : Detected matches

    _kp1 : Image keypoints

    _kp2 : Template keypoints
    """

    surf = cv2.xfeatures2d.SURF_create()

    kp1, des1 = surf.detectAndCompute(image, None)
    kp2, des2 = surf.detectAndCompute(temp, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    _matches = bf.match(des1, des2)
    _matches = sorted(_matches, key=lambda x: x.distance)

    # img3 = cv2.drawMatches(image, kp1, temp, kp2, _matches[:10], None, flags=2)
    # plt.imshow(img3)
    # plt.show()

    return _matches[:7], kp1, kp2


def pixel_positions(key1, key2, _matches):
    """
    Function extracts the spatial coordinates from the
    matched feature key points
    -----------------------------------------------------
    key1 : Keypoints for original image

    Key2 : Keypoints for template

    _matches : Matched feature points
    -----------------------------------------------------
    list_kp1 : Coordinates in original image

    list_kp2 : Coordinates in template

    matches : Matches for the refined coordinates
    """

    list_kp1 = []
    list_kp2 = []
    list_mat = []

    # For each match...
    for mat in _matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = key1[img1_idx].pt
        (x2, y2) = key2[img2_idx].pt

        # if (x2, y2) == (29.39104652404785, 26.495248794555664):
        list_kp1.append((x1, y1))
        list_kp2.append((x2, y2))
        list_mat.append(mat)

    return list_kp1, list_kp2, list_mat


def position_refinement(_loc, _kp1):
    """
    Function further filters and refines the detected
    pixel positions to obtain desired result
    -------------------------------------------------
    _loc : Points detected after template matching

    _kp1 : Keypoint coordinates after feature matching
    -------------------------------------------------
    refined : Final coordinates

    refined_keys : Final key point coordinates
    """

    points_list = []
    key_points = []
    _tol = 50

    for pts in zip(*_loc[::-1]):
        for pt1 in _kp1:
            if np.abs(pt1[0] - pts[0]) < _tol and np.abs(pt1[1] - pts[1] < _tol):
                points_list.append(pts)
                key_points.append(pt1)

    print(len(points_list))

    refined = []
    refined_key = []
    _tol = 5
    for ptr2 in range(0, len(points_list) - 1, 1):
        p1 = points_list[ptr2]
        p2 = points_list[ptr2 + 1]
        if len(refined) == 0:
            refined.append(p1)
            refined_key.append(key_points[ptr2])
        if np.abs(p2[0] - p1[0]) > _tol and np.abs(p2[1] - p1[1]) > _tol:
            # print("ptr1:", p1)
            # print("ptr2:", p2)
            if ptr2 not in refined:
                refined.append(p2)
                refined_key.append(key_points[ptr2+1])

    return refined, refined_key
