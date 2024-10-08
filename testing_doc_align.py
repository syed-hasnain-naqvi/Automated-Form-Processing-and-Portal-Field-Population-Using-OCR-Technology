# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:17:59 2024

@author: abc
"""
import doc_align
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_file = os.path.join(script_dir, '1009044688.pdf')
filename = os.path.splitext(os.path.basename(pdf_file))[0]
template_file = os.path.join(script_dir,'template.pdf')
output_pdf = os.path.join(script_dir, f'{filename}_output_aligned.pdf')

if not os.path.exists(pdf_file):
    print(f"PDF file does not exist: {pdf_file}")
if not os.path.exists(template_file):
    print(f"Template file does not exist: {template_file}")

try:
    output_pdf_path = doc_align.align_images_to_pdf(pdf_file, template_file)
    
    with open("debug_stage_2_output.pdf", "wb") as f:
        f.write(output_pdf_path)
    # print("Output PDF saved to ",output_pdf_path)
except Exception as e:
    print(f"An error occurred: {e}")