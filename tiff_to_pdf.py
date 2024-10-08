from PIL import Image
import os

def tiff_to_pdf(tiff_path, pdf_path):
    try:
        # Open the TIFF image
        with Image.open(tiff_path) as img:
            # Convert the image to RGB mode
            img = img.convert('RGB')
            # Save the image as a PDF
            img.save(pdf_path)
        print(f"Successfully converted {tiff_path} to {pdf_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
tiff_path = r'D:\my-work\Hasnain\FYP\Testing_more_dir\Testing_more_dir\sample1.tiff'
pdf_path =  r'D:\my-work\Hasnain\FYP\Testing_more_dir\Testing_more_dir\sample1.pdf'

tiff_to_pdf(tiff_path, pdf_path) 
