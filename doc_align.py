# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:17:09 2024

@author: syed.danish
"""
import cv2
import os
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from abc import ABC, abstractmethod
from PyPDF2 import PdfReader
import tempfile
from PIL import Image, ImageEnhance
import shutil
import io

script_dir = os.path.dirname(os.path.abspath(__file__))

class ImageAligner(ABC):
    def __init__(self, template: str) -> None:
        super().__init__()
        template_im: cv2.Mat = cv2.imread(template)
        self._template: cv2.Mat = cv2.cvtColor(template_im, cv2.COLOR_BGR2GRAY)

    @abstractmethod
    def align(self, query_image: cv2.Mat) -> cv2.Mat:
        pass

class SIFTAligner(ImageAligner):
    def __init__(self, template: str) -> None:
        super().__init__(template)
        self._h, self._w = self._template.shape
        self._sift = cv2.SIFT_create()
        self._kps_template, self._desc_template = self._sift.detectAndCompute(self._template, None)
        self._matcher = cv2.BFMatcher()

    def align(self, query_image: cv2.Mat) -> cv2.Mat:
        query = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        kps_query, desc_query = self._sift.detectAndCompute(query, None)
        matches = self._matcher.knnMatch(desc_query, self._desc_template, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:  # Adjusted ratio for more tolerance
                good.append(m)

        if len(good) < 4:
            raise ValueError(f"Not enough matches found to compute homography: {len(good)} matches")

        pts_template = np.zeros((len(good), 2), dtype="float")
        pts_query = np.zeros((len(good), 2), dtype="float")

        for i, m in enumerate(good):
            pts_query[i, :] = kps_query[m.queryIdx].pt
            pts_template[i, :] = self._kps_template[m.trainIdx].pt

        H, mask = cv2.findHomography(pts_query, pts_template, method=cv2.RANSAC)
        aligned = cv2.warpPerspective(query_image, H, (self._w, self._h))
        return aligned

def get_pdf_dimensions(pdf_path):
    reader = PdfReader(pdf_path)
    page = reader.pages[0]
    media_box = page.mediabox
    width = media_box.width
    height = media_box.height
    return width, height

def convert_pdf_to_images(pdf_file):
    return convert_from_path(pdf_file)

# def resize_and_save_images(images, size, prefix, temp_dir):
#     filenames = []
#     for i, image in enumerate(images):
#         image = np.array(image)
#         resized_image = cv2.resize(image, (1103, 1560))  # Correctly resize the image
#         resized_image = Image.fromarray(resized_image)
#         filename = os.path.join(temp_dir, f'{prefix}_{i}.jpg')
#         resized_image.save(filename, 'JPEG')
#         filenames.append(filename)
#     return filenames

def resize_and_save_images(images, size, prefix, temp_dir, format='JPEG', quality=95):
    
    filenames = []
    target_width, target_height = (1103, 1560)

    for i, image in enumerate(images):
        try:
            # Convert PIL Image to numpy array
            original_image = np.array(image)

            # # Optionally enhance image before resizing
            # image_enhanced = cv2.equalizeHist(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY))
            # image_enhanced = cv2.cvtColor(image_enhanced, cv2.COLOR_GRAY2RGB)

            # # Choose interpolation method based on resizing needs
            # if original_image.shape[1] > target_width or original_image.shape[0] > target_height:
            #     interpolation = cv2.INTER_AREA  # Better for reduction
            # else:
            #     interpolation = cv2.INTER_CUBIC  # Better for enlargement

            # Resize the image
            resized_image = cv2.resize(original_image, (target_width, target_height)) #, interpolation=interpolation)

            # Convert back to PIL Image for further enhancement
            resized_image_pil = Image.fromarray(resized_image)

            # Apply sharpening using PIL
            enhancer = ImageEnhance.Sharpness(resized_image_pil)
            final_image = enhancer.enhance(1.5)  # Factor >1 enhances sharpness, adjust as needed

            # Prepare the filename and save the image
            filename = os.path.join(temp_dir, f'{prefix}_{i}.{format.lower()}')
            final_image.save(filename, format, quality=quality)
            filenames.append(filename)

        except Exception as e:
            print(f'Failed to process image {i} with error: {e}')

    return filenames

def clean_temp_directory(temp_dir):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Deleted temporary directory: {temp_dir}")

def align_images_to_pdf(pdf_file, template_file):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_buffer = io.BytesIO()
        print("Starting PDF alignment process...")

        try:
            # Convert PDF to images
            pdf_images = convert_pdf_to_images(pdf_file)
            template_images = convert_pdf_to_images(template_file)
            
            if not pdf_images or not template_images:
                print("Error: PDF conversion to images failed.")
                return None

            width, height = get_pdf_dimensions(pdf_file)
            print(f"Dimensions Before Alignment = Width: {width}, Height: {height}")

            # Resize and save images to temporary directory
            pdf_filenames = resize_and_save_images(pdf_images, (794, 1123), 'image', temp_dir)
            template_filenames = resize_and_save_images(template_images, (794, 1123), 'template', temp_dir)

            if not pdf_filenames or not template_filenames:
                print("Error: Resizing or saving images failed.")
                return None

            # Align images using SIFT
            aligned_images = []
            for i, (image_path, template_path) in enumerate(zip(pdf_filenames, template_filenames)):
                print(f"Aligning image {i} using template {template_path}")
                sift_aligner = SIFTAligner(template_path)
                query_image = cv2.imread(image_path)
                
                if query_image is None:
                    print(f"Error: Failed to read image {image_path}")
                    continue

                try:
                    aligned_image = sift_aligner.align(query_image)
                    aligned_filename = os.path.join(temp_dir, f'aligned_image_{i}.jpg')
                    cv2.imwrite(aligned_filename, aligned_image)
                    aligned_images.append(aligned_filename)
                except ValueError as e:
                    print(f"Error aligning image {i}: {e}")
                    aligned_images.append(image_path)  # Use original if alignment fails

            if not aligned_images:
                print("Error: No aligned images found.")
                return None

            # Convert aligned images to a single PDF
            aligned_pil_images = [Image.open(img).convert('RGB') for img in aligned_images]
            aligned_pil_images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=aligned_pil_images[1:], resolution=100.0)
            pdf_buffer.seek(0)  # Ensure buffer pointer is at the beginning

            print("Saved aligned PDF to in-memory buffer")

            # Example: assuming get_pdf_dimensions can handle BytesIO
            width, height = get_pdf_dimensions(pdf_buffer)
            print(f"Dimensions After Alignment = Width: {width}, Height: {height}")

            output_pdf = pdf_buffer.getvalue()  # Get the byte data from buffer

            # Save the final PDF to disk for inspection
            with open("debug_output.pdf", "wb") as f:
                f.write(output_pdf)

            print("Saved debug output PDF to disk as debug_output.pdf")

        except Exception as e:
            print(f"An error occurred: {e}")
            output_pdf = None  # Return None if there is an error

        return output_pdf  # Return the PDF byte data


# if __name__ == "__main__":
#     pdf_file = 'E:/account_opening/code_base_001/zip/account_opening/code/pdf_files/1009034346.pdf'
#     template_file = 'E:/account_opening/code_base_001/zip/account_opening/code/pdf_files/template.pdf'
#     output_pdf = '1009034346_aligned_output.pdf'
#     align_images_to_pdf(pdf_file, template_file, output_pdf)
