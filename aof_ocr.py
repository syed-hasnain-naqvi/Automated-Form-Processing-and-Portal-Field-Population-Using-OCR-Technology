# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:17:08 2024

@author: umer
@author2: syed.danish
"""

from pdf2image import convert_from_path, convert_from_bytes
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image,ImageDraw,ImageFilter
import cv2
import json
import os
import io
import tempfile
import datetime
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import checkbox_detection
import doc_align
import re
import similarity

script_dir = os.path.dirname(os.path.abspath(__file__))
popplers_path = os.path.join(script_dir, 'poppler', 'Library','bin')
handwritten_model_dir = os.path.join(script_dir,'trocr-large-handwritten')

check_dir = os.path.join(script_dir,'exported_data','check_boxes')
select_dir = ""
print_dir = os.path.join(script_dir,'exported_data','printed_text')
hand_dir = os.path.join(script_dir,'exported_data','hand_text')
sign_dir = os.path.join(script_dir,'exported_data','sign_text')
#template_file = os.path.join(script_dir,'template_5_dir.pdf')



current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime("%Y-%m-%d")
current_year = datetime.datetime.now()
formatted_year = current_year.strftime("%Y")
annotation_id = 1
hand_list = []
hand_conf = []
check_dict = []
hand_list_gcp = []
print_list = []
multiple_hand = []
#images = []
pdf_name = ""
annotation_id = 0
ann_id = 1
current_image_index = 0
start_x = None
start_y = None
current_rectangle = None
bounding_boxes = {}
checkbox_dict = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

def load_handwritten_model(handwritten_model_dir):

    print(f"Using device: {device}")
    
    model = VisionEncoderDecoderModel.from_pretrained(handwritten_model_dir).to(device)
    processor = TrOCRProcessor.from_pretrained(handwritten_model_dir)
    return model, processor

def load_ocr():
    htw_model, htw_processor = load_handwritten_model(handwritten_model_dir)
    # Ensure both are loaded successfully
    if htw_model is None or htw_processor is None:
        print("Failed to load model or processor")
        return None, None
    
    return htw_model, htw_processor
    
def ocr(sharpened_image, model, processor):
    pixel_values = processor(images=sharpened_image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values, output_scores = True,  return_dict_in_generate=True, max_new_tokens=20)
    # text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    text = processor.batch_decode(generated_ids.sequences)
    probabilities = [torch.softmax(scores, dim=-1).max(dim=-1)[0] for scores in generated_ids.scores]
    mean_confidences = [scores.mean().item() for scores in probabilities]
    
    # Print predictions with their confidence scores
    for prediction, confidence in zip(text, mean_confidences):        
        print(f'Prediction: {prediction}, Confidence: {confidence:.2f}')            

    if confidence < 0.25:
        text = ""
    return text, confidence


def load_master(master_file):
    df_master = pd.read_excel(master_file)
    return df_master

def search_title(title_dict, title):
    if title in title_dict.keys():
        result = title_dict[title]
        return result
    else:
        print ('NA')
        return "NA"
    

def delete_excess_pages(pdf_path, max_pages=7):
    with open(pdf_path, 'rb') as f:
        pdf_data = f.read()
    try:
        # Ensure the input PDF data is in bytes
        if not isinstance(pdf_data, (bytes, bytearray)):
            raise ValueError("Input PDF data must be in bytes, not a string.")

        # Load the PDF from byte data
        pdf_stream = io.BytesIO(pdf_data)
        document = fitz.open(stream=pdf_stream, filetype="pdf")

        # Check if the PDF was read correctly
        if document.page_count == 0:
            raise ValueError("No pages found in PDF.")

        # Create a new PDF with only the first max_pages pages
        new_document = fitz.open()
        for i in range(min(max_pages, document.page_count)):
            new_document.insert_pdf(document, from_page=i, to_page=i)

        # Save the new PDF to a byte stream
        output_pdf = io.BytesIO()
        new_document.save(output_pdf)
        output_pdf.seek(0)

        return output_pdf.getvalue()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

import pandas as pd
from fuzzywuzzy import process

# Load the mappings from the Excel file
def load_country_mappings(excel_file, des, code):
    df = pd.read_excel(excel_file)
    # Assuming the columns are named 'Country' and 'NumericValue'
    country_mapping = dict(zip(df[des], df[code]))
    return country_mapping

# Function to correct a misspelled country name and get its numeric value
def get_mapped_value(input_name, country_mapping):
    # Extract the country names and their numeric values
    if input_name != "":
        countries = list(country_mapping.keys())
        
        # Find the closest match in the list of countries
        best_match, match_score = process.extractOne(input_name, countries)
        
        # If the similarity score is above a threshold, get the mapped value
        if match_score > 70:
            return country_mapping[best_match]
        else:
            return "PK"  # or handle the case where no suitable match is found
    else:
        return "PK"
def mappings(df):
    excel_file = r'map.xlsx'
    country_mapping_nat = load_country_mappings(excel_file, "DES1", "AC1")

    misspelled_country = [3,15,27,38]
    for index in misspelled_country:
        mapped_value = get_mapped_value(df.extracted[index], country_mapping_nat)
        df.extracted[index] = mapped_value
    country_mapping_occ = load_country_mappings(excel_file, "occ", "occ_code")
    mapped_value_occ = get_mapped_value(df.extracted[6], country_mapping_occ)
    df.extracted[6] = mapped_value_occ



def upload_pdf(pdf_path):
    if not pdf_path:
        return

    filename = os.path.basename(pdf_path)
    root, extension = os.path.splitext(filename)
    output_pdf_path = None  # Initialize the variable
    
    #pdf_img = doc_align.convert_pdf_to_images(pdf_path)
    img = convert_from_path(pdf_path)

    #difference = similarity.get_director(pdf_path, 3,4)
    print("----------")
    print(len(img))

    if len(img) == 7:
        template_file = os.path.join(script_dir,'template_3_dir.pdf')
        dirs = 3
        print("----------------3---------------")
    elif similarity.get_director(pdf_path, 3,5) < 900 and len(img) == 9:
        template_file = os.path.join(script_dir,'template_5_dir.pdf')
        dirs = 5
        print("----------------5---------------")
    elif  similarity.get_director(pdf_path, 3,7) < 900 and len(img) == 11:
        template_file = os.path.join(script_dir,'template_7_dir.pdf')
        dirs = 7
        print("----------------7---------------")
    elif  similarity.get_director(pdf_path, 3,9) < 900 and len(img) == 13:
        template_file = os.path.join(script_dir,'template_9_dir.pdf')
        dirs = 9
        print("----------------9---------------")
    elif  similarity.get_director(pdf_path, 3,11) < 900 and len(img) == 15:
        template_file = os.path.join(script_dir,'template_11_dir.pdf')
        dirs = 11
        print("----------------11---------------")
    else:
        excess_deleted = delete_excess_pages(pdf_path)
        with open(pdf_path, 'wb') as f:
            f.write(excess_deleted)
        print(len(convert_from_path(pdf_path)))
        template_file = os.path.join(script_dir,'template_3_dir.pdf')
        dirs = 3
        

        
        
    try:
        output_pdf_path = doc_align.align_images_to_pdf(pdf_path, template_file)
        # print(f"Output PDF saved to {output_pdf_path}")
    except Exception as e:
        print(f"An error occurred while aligning images to PDF: {e}")
    
    if output_pdf_path:  # Check if output_pdf_path was successfully assigned
        try:
            with open("output_pdf_path.pdf", "wb") as f:
                f.write(output_pdf_path)            
            image_list = convert_from_bytes(output_pdf_path)
            return image_list, dirs
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            print("Error loading PDF:", str(e))
    else:
        print("No output PDF path available to convert.")

def flatten_and_clean(item):
    # Ensure the item is a string
    if not isinstance(item, str):
        item = str(item)    
    # Replace the '</s>' tags in the string
    return item.replace('</s>', '').strip() if item else None

def clean_string(s):
    if not isinstance(s, str):
        s = str(s)  # Convert non-string items to string
    cleaned_s = re.sub(r'^[^A-Za-z]+|[^A-Za-z]+$', '', s)
    return re.sub(r'[\[\]]', '', cleaned_s)

def clean_number(s):
    if not isinstance(s, str):
        s = str(s)  # Convert non-string items to string
    cleaned_s = re.sub(r'^[^0-9]+|[^0-9]+$', '', s)
    return re.sub(r'[\[\]]', '', cleaned_s)

def extract_middle_part(title):
    # Remove the 'check_' prefix
    parts = title[len('check_'):].split('_')
    
    # Remove the last part
    parts = parts[:-1]
    
    # Capitalize the first letter of each part and join them with spaces
    middle_part = ' '.join([part.capitalize() for part in parts])
    
    return middle_part

    
def upload_json(images, df_master, annotation_path, htw_model, htw_processor, title_mapping):
    temp_dir = tempfile.mkdtemp()
    saved_image_paths = []
    cropped_images = []
    if not annotation_path: return
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    annotations = data.get('annotations', [])
    hand_text = ""
    printed_text = ""
    check_box_text = ""
    df = pd.DataFrame(columns=['Title', 'ExtractedData'])
    ann_id = 1
    cropped_image = ""
    j = 0
    for i, annotation in enumerate(annotations):
        title = annotation['title']
        image_id = annotation['image_id']
        area = annotation['area']
        bbox_data = annotation['bbox']        
        image_path = os.path.join(temp_dir, f'image_{0}.png')

        images[image_id-1].save(image_path, 'PNG')
        saved_image_paths.append(image_path)
        pillow_images = [Image.open(image_path) for image_path in saved_image_paths]
        
        draw = ImageDraw.Draw(pillow_images[0])
        start_x, start_y, end_x, end_y = bbox_data
        draw.rectangle([start_x, start_y, end_x, end_y], outline="red", width=2)
        cropped_region = pillow_images[0].crop((start_x, start_y, end_x, end_y))
        sharpened_image = cropped_region.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        cropped_images.append(sharpened_image)
        title_dict = df_master.set_index('JSON Title')['Title'].to_dict()
        checkbox_status = "False"
        if "hand" in title:            
            select_dir = hand_dir
            hand_text, hand_text_confidence = ocr(sharpened_image,htw_model, htw_processor)
            print("Title: ",title)            
            clean_hand_text = flatten_and_clean(hand_text)
            if "text" in title:
                clean_hand_text = clean_string(clean_hand_text)
            elif "numeric" in title:
                clean_hand_text = clean_number(clean_hand_text)
            print("Handwritten Text: ", clean_hand_text)
            hand_list.append(clean_hand_text)

        if "print" in title:
            select_dir = print_dir
            printed_text = search_title(title_dict, title) #ocr(sharpened_image, p_model, p_processor) 
            print_list.append(printed_text)
        elif 'check' in title:            
            select_dir = check_dir
            print("Check Title: ", title)
            check_box_text = search_title(title_dict, title)
            
            stats, labels, imagee = checkbox_detection.detect_box(sharpened_image)
            for x, y, w, h, area in stats[2:]:
                cv2.rectangle(imagee, (x, y), (x+w, y+h), (0, 255, 0), 1)
            bounding_boxes = stats
            cropped_image = checkbox_detection.crop_and_append_images(imagee, bounding_boxes)
            checkbox_status = checkbox_detection.check_status(cropped_image)
            print("Check box Text: ", check_box_text)
            print("\nCheck Box Status: ", checkbox_status)
            desired_val='Check None'
            if checkbox_status=="True":
                desired_val= check_box_text
                hand_list.append(desired_val)
                print_list.append(extract_middle_part(title))
                j+=1
                print(" check approved: ",j)
        elif "sign" in title:
            select_dir = sign_dir
            sharpened_image.save(f'{select_dir}{pdf_name}_{current_image_index}_{ann_id}.jpg')
                
        for i, cropped_image in enumerate(cropped_images):
            cropped_image.save(f'{select_dir}{pdf_name}_{current_image_index}_{ann_id}.jpg')
            ann_id = ann_id + 1
        cropped_images.clear()
        
    print("Handwritten List: ", len(hand_list))
    print("Handwritten COnfidence List: ", len(hand_conf))
    print("Printed Title List: ", len(print_list))
    print("Handwritten text List", hand_list)
    print("\nPrinted text List", print_list)
    print("\nNo of checks approved: ",j)
    modified_list = [item[0] if isinstance(item, (list, tuple)) else item for item in hand_list]
    print("\n\nHandwritten text Modified List", modified_list)
    df = pd.DataFrame(print_list, columns=['title'])
    df['extracted'] = modified_list
    df['extracted'] = df['extracted'].str.replace(r"[(.'\")]", '', regex=True)
    print("The Dataframe: ",df)
    return df

