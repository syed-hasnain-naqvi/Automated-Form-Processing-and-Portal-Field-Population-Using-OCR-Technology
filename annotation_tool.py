import tkinter as tk
from tkinter import filedialog,Canvas,Scrollbar,simpledialog, messagebox
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image, ImageTk,ImageDraw,ImageFont,ImageFilter
import json
import os
import numpy as np
import tempfile
import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
current_datetime = datetime.datetime.now()
formatted_date = current_datetime.strftime("%Y-%m-%d")
current_year = datetime.datetime.now()
formatted_year = current_year.strftime("%Y")
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
class PDFImageApp:

    # Initialize COCO JSON data structure
    def __init__(self, root):
        root.title("PDF Annotation Tool")
        root.geometry("1300x700")
        self.coco_data = {
            "info": {
                "description": "PDF Annotation",
                "year": formatted_year,
                "contributor": "Syed Muhammad Danish",
                "date_created": formatted_date
            },
            "images": [],
            "annotations": [],
            "categories": []  # Modify to define categories as needed
        }

        # Annotation ID counter
        self.annotation_id = 1

        self.Heading = tk.Label(root, text="PDF Annotation Tool", font=("Times New Roman", 32, "bold"))
        self.Heading.place(relx=0.5, y=20, anchor='n')

        self.instruction_label = tk.Label(root, text="Start here...", font=("Arial", 16))
        self.instruction_label.place(x=100, y=20)
        
        self.upload_pdf_button = tk.Button(root, text="Upload PDF", command=self.upload_pdf, font=("Arial", 14), bg = "#FF0055", fg="white")
        self.upload_pdf_button.place(x=100, y=50)
        
        self.upload_json_button = tk.Button(root, text="Upload JSON", command=self.upload_json, font=("Arial", 14), state=tk.DISABLED, bg = "#FF2D00", fg="white")
        self.upload_json_button.place(x=250, y=50)

        self.canvas = tk.Canvas(root, bg="white", width=1400, height=850)
        self.canvas.place(x=100, y=100)
        
        self.prev_button = tk.Button(root, text="<--Previous", command=self.prev_image, font=("Arial", 14), state=tk.DISABLED, bg="#907985", fg="white")
        self.prev_button.place(x=100, y=960)
        
        self.next_button = tk.Button(root, text="Next    -->", command=self.next_image, font=("Arial", 14), state=tk.DISABLED, bg="#907985", fg="white")
        self.next_button.place(x=250, y=960)

        # self.apply_format_button = tk.Button(root, text="Apply Formatting", command=self.apply_formatting, font=("Arial", 14))
        # self.apply_format_button.place(x=1400, y=960)

        self.scroll_frame = tk.Frame(root)
        self.scroll_frame.place(x=100, y=100, width=1600, height=850)

        self.v_scroll = Scrollbar(self.scroll_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.h_scroll = Scrollbar(self.scroll_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = Canvas(self.scroll_frame, bg="lightgrey", width=1600, height=850,
                             yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # img = tk.PhotoImage(file='D:\hello\OCR_acc_opening\alpha_logo.jpg')  # Update with the path to your image
        # self.canvas.create_image(0, 0, anchor='nw', image=img)        
        # self.canvas.place(x=300. y=70)

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)


        self.save_button = tk.Button(root, text="Save", command=self.save_data, font=("Arial", 14), bg="#907985", fg="white")
        self.save_button.place(x=1600, y=960)


        self.images = []
        self.pdf_name = ""
        self.annotation_id = 0
        self.ann_id = 1
        self.current_image_index = 0
        self.start_x = None
        self.start_y = None
        self.current_rectangle = None
        self.bounding_boxes = {}

        self.canvas.bind("<Button-1>", self.start_draw_bbox)
        self.canvas.bind("<B1-Motion>", self.dragging)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw_bbox)


    def save_data(self):
        dirpath = filedialog.askdirectory(title="Select a Directory to Save COCO JSON")
        if not dirpath:
            return

        for idx, img in enumerate(self.images):
            img_path = os.path.join(dirpath, f"original_image_{idx}.png")
            img.save(img_path, 'PNG')

            # Add image to COCO JSON
            image_info = {
                "id": idx + 1,
                "file_name": f"original_image_{idx}.png",
                "width": img.width,
                "height": img.height
            }
            self.coco_data["images"].append(image_info)

        # Iterate through bounding boxes and add them to COCO JSON
        for name, bbox_data_list in self.bounding_boxes.items():
            for bbox_data in bbox_data_list:
                annotation = {
                    "id": self.annotation_id,
                    "title": bbox_data["title"],
                    "image_id": bbox_data["image_index"] + 1,
                    "category_id": 1,  # Modify to your category ID
                    "segmentation": [],
                    "area": float((bbox_data["coords"][2] - bbox_data["coords"][0]) * (bbox_data["coords"][3] - bbox_data["coords"][1])),
                    "bbox": bbox_data["coords"],
                    "iscrowd": 0
                }
                self.coco_data["annotations"].append(annotation)
                self.annotation_id += 1

        # Save COCO JSON file
        json_path = os.path.join(dirpath, "coco_annotations.json")
        with open(json_path, 'w') as f:
            json.dump(self.coco_data, f, indent=4)


    def apply_formatting(self):


        dirpath = filedialog.askdirectory(title="Select a Directory to Save PDFs")
        if not dirpath: return

        max_texts = max([len(formatting['texts']) for formatting in self.bbox_formatting.values()])

        for text_idx in range(max_texts):
            modified_images = [img.copy() for img in self.images]

            for name, bbox_data_list in self.bounding_boxes.items():
                formatting = self.bbox_formatting.get(name, {})
                texts = formatting.get("texts", [])
                if text_idx >= len(texts): continue
                text = texts[text_idx]
                for bbox_data in bbox_data_list:
                    img = modified_images[bbox_data["image_index"]]
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(bbox_data['coords'], fill=formatting.get("background_color"))
                    font_size = formatting.get("font", {}).get("size", 20)
                    font_color = formatting.get("font", {}).get("color", "black")
                    font_style = formatting.get("font", {}).get("style", "normal")
                    
                    if font_style == "bold":
                            font = ImageFont.truetype("d:/times new roman bold.ttf", font_size)
                    elif font_style == "italic":
                            font = ImageFont.truetype("d:/times new roman italic.ttf", font_size)
                    elif font_style == "default":
                            font=ImageFont.truetype("d:/noto/NotoSans-Bold.ttf")
                    else:
                            font = ImageFont.truetype("d:/times new roman.ttf", font_size)
                    text_x = bbox_data['coords'][0]
                    text_y = bbox_data['coords'][1]
                    draw.text((text_x, text_y), text, fill=font_color, font=font)

            pdf_filename = f"modified_{text_idx + 1}.pdf"
            pdf_path = os.path.join(dirpath, pdf_filename)
            modified_images[0].save(pdf_path, save_all=True, append_images=modified_images[1:])

        self.display_image(self.current_image_index)

    def start_draw_bbox(self, event):
        self.canvas.delete(self.current_rectangle)
        self.current_rectangle = None
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

    def dragging(self, event):
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if self.current_rectangle:
            self.canvas.delete(self.current_rectangle)
        self.current_rectangle = self.canvas.create_rectangle(self.start_x, self.start_y, x, y, outline='red')

    def end_draw_bbox(self, event):
        bbox_name = simpledialog.askstring("Bounding Box Name", "Name For Field:")
        if bbox_name:
            bbox_data = {
                "coords": (self.start_x, self.start_y, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)),
                "image_index": self.current_image_index,
                "title": bbox_name
            }
            print(bbox_data)
            if bbox_name in self.bounding_boxes:
                self.bounding_boxes[bbox_name].append(bbox_data)
            else:
                self.bounding_boxes[bbox_name] = [bbox_data]
            print("saving annotation")

    def draw_bounding_boxes(self):
        # Create a blank image
        # print(bbox_data["coords"])
        image = Image.new("RGB", (500, 500), "white")
        draw = ImageDraw.Draw(image)

        for bbox_name, bbox_list in self.bounding_boxes.items():
            for bbox_data in bbox_list:
                start_x, start_y, end_x, end_y = bbox_data["coords"]
                draw.rectangle([start_x, start_y, end_x, end_y], outline="red", width=2)

        # Display the image on the canvas
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.image_tk, anchor="nw")
        

    def save_annotated_image(self, filename):
        temp_dir = tempfile.mkdtemp()
        saved_image_paths = []
        cropped_images = []
        
        image_path = os.path.join(temp_dir, f'image_{0}.png')
        self.images[self.current_image_index].save(image_path, 'PNG')
        saved_image_paths.append(image_path)
        pillow_images = [Image.open(image_path) for image_path in saved_image_paths]

        draw = ImageDraw.Draw(pillow_images[0])

        for idx, (bbox_name, bbox_list) in enumerate(self.bounding_boxes.items()):
            if bbox_list:  # Check if the list is not empty
                if idx == len(self.bounding_boxes) - 1:
                    # Draw the bounding box
                    bbox_data = bbox_list[-1]  # Get the last bbox_data in the list
                    start_x, start_y, end_x, end_y = bbox_data["coords"]
                    draw.rectangle([start_x, start_y, end_x, end_y], outline="red", width=2)

                    # Crop the region enclosed by the bounding box
                    cropped_region = pillow_images[0].crop((start_x, start_y, end_x, end_y))

                    # Append the cropped region to the list
                    cropped_images.append(cropped_region)

                print("saving images")

    def apply_annotation():
        print("anottating")

    def upload_pdf(self):
        filepath = filedialog.askopenfilename(title="Select a PDF", filetypes=[("PDF files", "*.pdf")])
        if not filepath: 
            return
        #self.images = convert_from_path(filepath)
        filename = os.path.basename(filepath)
        root, extension = os.path.splitext(filename)
        self.pdf_name = root
        try:
            self.images = convert_from_path(filepath, poppler_path=os.path.join(script_dir,'poppler','bin'))
            self.ann_id = 1
            # Rest of your code
        except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
            print("Error loading PDF:", str(e))
        
        self.display_image(0)
        if len(self.images) > 1:
            self.next_button.config(state=tk.NORMAL)
        self.upload_json_button.config(state=tk.NORMAL)

    def display_image(self, index):
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(self.images[index])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.current_image_index)
            if self.current_image_index == 0:
                self.prev_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.NORMAL)

    def next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.display_image(self.current_image_index)
            if self.current_image_index == len(self.images) - 1:
                self.next_button.config(state=tk.DISABLED)
        self.prev_button.config(state=tk.NORMAL)

    def upload_json(self):
        print("Annotation started")
        temp_dir = tempfile.mkdtemp()
        saved_image_paths = []
        cropped_images = []
        hand_dir = os.path.join(script_dir,'exported_data','hand_text')
        select_dir = ""
        print_dir = os.path.join(script_dir,'exported_data','printed_test')
        check_dir = os.path.join(script_dir,'exported_data''check_boxes')
        
        filepath = filedialog.askopenfilename(title="Select a JSON", filetypes=[("JSON files", "*.json")])
        if not filepath: return
        with open(filepath, 'r') as f:
            data = json.load(f)
        annotations = data.get('annotations', [])
        print("Working........")
        print(annotations)
        for annotation in annotations:
            print("Working")
            title = annotation['title']
            image_id = annotation['image_id']

            bbox_data = annotation['bbox']
            
            image_path = os.path.join(temp_dir, f'image_{0}.png')
            self.images[image_id-1].save(image_path, 'PNG')
            saved_image_paths.append(image_path)
            pillow_images = [Image.open(image_path) for image_path in saved_image_paths]

            draw = ImageDraw.Draw(pillow_images[0])
            start_x, start_y, end_x, end_y = bbox_data
            draw.rectangle([start_x, start_y, end_x, end_y], outline="red", width=2)
            cropped_region = pillow_images[0].crop((start_x, start_y, end_x, end_y))
            sharpened_image = cropped_region.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            cropped_images.append(sharpened_image)
            print("saving images")
            print(title)
            if "hand" in title:
                select_dir = hand_dir
            elif "print" in title:
                select_dir = print_dir
            elif 'check' in title:
                select_dir = check_dir

            for i, cropped_image in enumerate(cropped_images):
                cropped_image.save(f'{select_dir}{self.pdf_name}_{image_id}_{self.ann_id}.jpg')                
                self.ann_id = self.ann_id + 1
            cropped_images.clear()
        messagebox.showinfo( "Alert!", "Annotation Completed! Check Folders")
        print("Completed")


        
    


root = tk.Tk()
app = PDFImageApp(root)

root.mainloop()