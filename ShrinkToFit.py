import os
import os.path
from PIL import Image
import glob

def jpgToTIFF(folder):
    os.chdir(folder)

    for infile in glob.glob("*.jpg"):
        file, ext = os.path.splitext(infile)
        print(infile)
        im = Image.open(infile)
        ensure_dir(folder+"/tiff/"+file+".tiff")
        im.save(folder+"/tiff/"+file+".tiff", 'TIFF')

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
subfolders = [f.path for f in os.scandir(r"C:\Users\sardo\Documents\GitHub\WNet\BSR\BSDS500\data\images") if f.is_dir() ]
for folder in subfolders:
    print(folder)
    jpgToTIFF(folder)

