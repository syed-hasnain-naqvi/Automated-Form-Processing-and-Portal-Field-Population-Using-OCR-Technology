# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:59:13 2024

@author: syed.danish
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

plot_flag=False
save_output=True
cropped_images_dictionary = {}
def plot(image,cmap=None):
    plt.figure(figsize=(15,15))
    plt.imshow(image,cmap=cmap) 
    
def imshow_components(labels):
    label_hue = np.uint8(179*labels/np.max(labels)) ### creating a hsv image, with a unique hue value for each label
    empty_channel = 255*np.ones_like(label_hue)     ### making saturation and volume to be 255
    labeled_img = cv2.merge([label_hue, empty_channel, empty_channel])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR) ### converting the hsv image to BGR image
    labeled_img[label_hue==0] = 0
    return labeled_img ### returning the color image for visualising Connected Componenets

def detect_box(image,line_min_width=15):
    if image is not None and not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.uint8)
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(gray_scale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,5)
    kernal6h = np.ones((1,line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width,1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final= cv2.bitwise_or(img_bin_h, img_bin_v)
    final_kernel = np.ones((3,3), np.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)
    numLabels, labels, stats, centroids  = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    return stats,labels,image        


def crop_and_append_images(image, bounding_boxes, index=0):
    if image is None or index >= len(bounding_boxes): # Check if image is None or if index is out of range
        print("Error: Image not found or index out of range.")
        return
    
    for i, (x, y, w, h, area) in enumerate(bounding_boxes): # Iterate over the bounding boxes and crop the image
        # print("Area: ", area)
        if area in range(200, 601):            
            cropped_image = image[y:y+h, x:x+w]         # Crop the image
        else:
            cropped_image = None
    return cropped_image
    
    

def calculate_black_white_percentage(image):
    black_count = np.sum(image <= 200) # Count black pixels (pixel value == 0)
    white_count = np.sum(image > 200)  # Count white pixels (pixel value == 255)
    total_pixels = image.size          # Total number of pixels
    black_percentage = (black_count / total_pixels) * 100 # Calculate percentages
    white_percentage = (white_count / total_pixels) * 100
    return black_percentage, white_percentage

def check_status(image_file):    
    status = "False"
    if image_file is None:
        return status
    else:
        aligned_sift_copy_6 = image_file.copy()
        regions = []
        SELECTION_COUNT_THRESHOLD = 0
        im = aligned_sift_copy_6
    
        #preprocessing
        laplacian_var = cv2.Laplacian(im, cv2.CV_64F).var()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        AREA_THRESHOLD = 10
        for c in cnts:
            area = cv2.contourArea(c)
            if area < AREA_THRESHOLD:
                cv2.drawContours(thresh, [c], -1, 0, -1)
        repair_kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
        repair_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, repair_kernel_horizontal, iterations=2)
        horizontal_lines = cv2.HoughLinesP(repair_horizontal, 1, np.pi/180, 20, 10, 10)
        im_copy_horiz = im.copy()
        if horizontal_lines is not None:
            for line in horizontal_lines:
                for x1, y1, x2, y2 in line:
                    theta = np.arctan((y2 - y1) / (x2 - x1 + 1e-8)) * 180 / np.pi
                    if (theta > -6 and theta < 6):
                        cv2.line(thresh, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.line(im_copy_horiz, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
        repair_kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))
        repair_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, repair_kernel_vertical, iterations=1)
        vertical_edges = cv2.Canny(repair_vertical, 50, 200, apertureSize=3)
        vertical_lines = cv2.HoughLinesP(repair_vertical, 1, np.pi/180, 12, 40, 4)
        im_copy_vert = im.copy()
    
        if vertical_lines is not None:
            for line in vertical_lines:
                for x1, y1, x2, y2 in line:
                    theta = np.abs(np.arctan((y2 - y1)/(x2 - x1 + 1e-8)) * 180 / np.pi)
                    # print(f'Vertical: Entity: {sample_region["type"]}, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, theta: {theta}')
                    if (theta > 84 and theta < 96):
                        cv2.line(thresh, (x1, y1), (x2, y2), (0, 0, 0), 2)
                        cv2.line(im_copy_vert, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Cleaning the final image noise with contours
        cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        AREA_THRESHOLD = 10
        for c in cnts:
            area = cv2.contourArea(c)
            if area < AREA_THRESHOLD:
                cv2.drawContours(thresh, [c], -1, 0, -1)
        non_black_count = cv2.countNonZero(thresh)
        black_percentage, white_percentage = calculate_black_white_percentage(thresh)
        Check = False        
        if white_percentage >= 20:
            Check = True
            status = "True"
            
        print(f"Black: {black_percentage}%, White: {white_percentage}% Check: {Check}")        
        print(f" Check: {Check}")
        return status
        