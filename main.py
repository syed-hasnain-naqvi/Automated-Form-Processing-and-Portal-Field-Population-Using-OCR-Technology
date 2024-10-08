# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:08:43 2024

@author: abc
"""
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd
import requests
from bs4 import BeautifulSoup
import aof_ocr


# --------------------------------change by hasnain
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Now import TensorFlow and other libraries
import tensorflow as tf
# -------------------------------change by hasnain

# import doc_align

pdf_var = ""
script_dir = os.path.dirname(os.path.abspath(__file__))
#template_pdf = os.path.join(script_dir,'template_5_dir.pdf')
output_pdf = os.path.join(script_dir,'output','output.pdf')
#annotation_path = os.path.join(script_dir, 'aof_json', 'coco_annotations_5.json')

def delete_all_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("The specified folder does not exist.")
        return
    
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Loop through the files and delete each one
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global pdf_var
        if event.is_directory:
            return
        print(f"File modified: {event.src_path}")
        pdf_var = event.src_path

    def on_created(self, event):
        global pdf_var
        if event.is_directory:
            return
        elif event.src_path.endswith('.pdf'):
            file_name = os.path.basename(event.src_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            print(f"New file detected: {file_name}")
            pdf_var = event.src_path
            process_pdf(pdf_var, file_name_without_ext)

def process_pdf(pdf_path, file_name_without_ext):
    print(f"Processing PDF: {pdf_path}")
    # output_pdf_path = doc_align.align_images_to_pdf(pdf_path, template_pdf, output_pdf)
    # print("\nOutput pdf path",pdf_path)
    images, dirs = aof_ocr.upload_pdf(pdf_path)
    print("Starting Extraction")
    if dirs == 3:
        annotation_path = os.path.join(script_dir, 'aof_json', 'coco_annotations_3.json')
        df_file = os.path.join(current_directory, 'aof_master_file_3_dir.xlsx')
    elif dirs == 5:
        annotation_path = os.path.join(script_dir, 'aof_json', 'coco_annotations_5.json')
        df_file = os.path.join(current_directory, 'aof_master_file_5_dir.xlsx')
    elif dirs == 7:
        annotation_path = os.path.join(script_dir, 'aof_json', 'coco_annotations_7.json')
        df_file = os.path.join(current_directory, 'aof_master_file_7_dir.xlsx')
    elif dirs == 9:
        annotation_path = os.path.join(script_dir, 'aof_json', 'coco_annotations_9.json')
        df_file = os.path.join(current_directory, 'aof_master_file_9_dir.xlsx')
    elif dirs == 11:
        annotation_path = os.path.join(script_dir, 'aof_json', 'coco_annotations_11.json')
        df_file = os.path.join(current_directory, 'aof_master_file_11_dir.xlsx')
    
    df_master = pd.read_excel(df_file)
    title_mapping = dict(zip(df_master['JSON Title'], df_master['Title']))
    df = pd.DataFrame(columns=(['title','extracted']))
    df = aof_ocr.upload_json(images, df_master, annotation_path, htw_model, htw_processor, title_mapping)
    output_excel = f'{file_name_without_ext}_data_output.xlsx'
    
    df.to_excel(output_excel, index=False)
    print("Completed")
    aof_ocr.mappings(df)
    delete_all_files_in_folder((os.path.join(script_dir, 'pdf_files')))
    # print("\nService Body creation started")
    # soap_body = create_soap_request(df)
    # print("\nService Body created")
    print("\nService call initiated\n")
    send_soap_request(df)
    print("\nService call ended\n Thank you for using our service\n")        
    print("-------------")

# def create_soap_request(df):
#     soap_body_template = """
#     <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:rdab="spyne.examples.hello.soap">
#        <soapenv:Header/>
#        <soapenv:Body>
#           <rdab:insert_record>
#               {Content}
#           </rdab:insert_record>
#        </soapenv:Body>
#     </soapenv:Envelope>
#     """
#     content = ""
#     for index, row in df.iterrows():
#         content += f"<rdab:{row['title']}>{row['extracted']}</rdab:{row['title']}>"
    
#     return soap_body_template.format(Content=content)  # Use Content as the placeholder

# def send_soap_request(soap_body):
#     url = 'http://127.0.0.1:8080/'  # Your local SOAP server URL
#     headers = {
#         'Content-Type': 'text/xml; charset=utf-8',
#         'SOAPAction': 'spyne.examples.hello.soap/insert_record'  # Your SOAP action
#     }
#     response = requests.post(url, data=soap_body, headers=headers)
    
#     if response.status_code == 200:
#         # Parse the response
#         response_soup = BeautifulSoup(response.content, 'lxml')
#         code = response_soup.find('code').text
#         message = response_soup.find('message').text
        
#         if code == '00':
#             print(f"Success: {message}")
#         elif code in ['-1', '99']:
#             print(f"Failure: {message}")
#         else:
#             print("Unexpected response code")
#     else:
#         print(f"HTTP Error: {response.status_code}")
def create_soap_request(df):
    soap_body_template = """
    <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:rdab="spyne.examples.hello.soap">
       <soapenv:Header/>
       <soapenv:Body>
          <rdab:insert_record>
              <rdab:record>
                  {Content}
              </rdab:record>
          </rdab:insert_record>
       </soapenv:Body>
    </soapenv:Envelope>
    """
    
    content = ""
    for index, row in df.iterrows():
        for col in df.columns:
            content += f"<rdab:{col}>{row[col]}</rdab:{col}>"
        break  # Assuming you want to process only the first row
    
    soap_body = soap_body_template.format(Content=content)
    print(soap_body)
    return soap_body

def send_soap_request(df):
    url = 'http://127.0.0.1:8080/'  # Your local SOAP server URL
    
    headers = {
        'Content-Type': 'text/xml; charset=utf-8',
        'SOAPAction': 'spyne.examples.hello.soap/insert_record'  # Your SOAP action
    }
    
    soap_body = create_soap_request(df)
    # print("\nThe Soap Body: \n", soap_body)
    response = requests.post(url, data=soap_body, headers=headers)
    
    if response.status_code == 200:
        # Parse the response
        response_soup = BeautifulSoup(response.content, 'xml')
        code_element = response_soup.find('code').text
        if code_element:
            code = code_element.text
            message = response_soup.find('message').text if response_soup.find('message') else "No message"
            if code == '200' or code == '00':
                print(f"Success: {message}")
            elif code == '99' or code == '500':
                print("T24 server down")
            elif code == '-1':
                print(f"Failure: {message}")
            else:   
                print("Unexpected response code")
        else:
            print("Response does not contain a code element.")
    else:
        print(f"HTTP Error: {response.status_code}")
        
        


if __name__ == "__main__":
    path = os.path.join(script_dir, "pdf_files")
    current_directory = os.getcwd()
    images = []

    if not os.path.exists(path):
        os.makedirs(path)
    
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    print("Program started")
    print("Model Initiating")
    htw_model, htw_processor = aof_ocr.load_ocr()
    print("Model Loaded")
    print("Loading master data")


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
 