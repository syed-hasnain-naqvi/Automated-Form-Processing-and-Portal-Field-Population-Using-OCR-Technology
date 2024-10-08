import os
import shutil
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def tiff_to_pdf(tiff_path, pdf_path):
    try:
        with Image.open(tiff_path) as img:
            imgs = []
            for i in range(img.n_frames):
                img.seek(i)
                imgs.append(img.convert('RGB'))
            imgs[0].save(pdf_path, save_all=True, append_images=imgs[1:])
        print(f"Successfully converted {tiff_path} to {pdf_path}")
    except Exception as e:
        print(f"An error occurred converting TIFF to PDF: {e}")

class MyHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        file_root, file_ext = os.path.splitext(file_name)
        
        if file_ext.lower() == '.tiff':
            pdf_path = os.path.join(r"D:\my-work\Hasnain\FYP\Testing_more_dir\Testing_more_dir\pdf_files\pdf_files", f"{file_root}.pdf")
            tiff_to_pdf(file_path, pdf_path)

path_to_watch = r"D:\my-work\Hasnain\FYP\Testing_more_dir\Testing_more_dir\pdf_files\sample_Rosetta_folder"
event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path_to_watch, recursive=False)
observer.start()

print("Monitoring for new .tiff files. Press Ctrl+C to exit.")
try:
    while True:
        pass
except KeyboardInterrupt:
    observer.stop()

observer.join()