import fitz
import os
import base64
from PIL import Image
import io
import uuid

def encodeImage(path: str):
    with open(path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def parse(path: str):
    """
    Parse the PDF at `path`.
    Returns: (list_of_text_blocks, list_of_image_paths)
    Each call uses a unique temporary directory for extracted images to avoid collisions.
    """
    print(f'Parsing {path}')
    imageOut = os.path.join("temp_images", str(uuid.uuid4()))
    os.makedirs(imageOut, exist_ok=True)

    doc = fitz.open(path)

    all_text = []
    image_paths = []

    try:
        for pageNum, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                all_text.append(page_text)
            try:
                imageList = page.get_images(full=True)
            except Exception as e:
                print(f"Warning: couldn't get images for page {pageNum+1}: {e}")
                imageList = []

            for imgIndex, img in enumerate(imageList):
                try:
                    xref = img[0]
                    base = doc.extract_image(xref)
                    imgBytes = base.get('image')
                    imgExt = base.get('ext', 'png')
                    imgPath = os.path.join(imageOut, f'page{pageNum + 1}_img{imgIndex + 1}.{imgExt}')

                    with open(imgPath, 'wb') as imgFile:
                        imgFile.write(imgBytes)

                    image_paths.append(imgPath)
                except Exception as e:
                    print(f"Warning: failed to extract image on page {pageNum+1} image #{imgIndex+1}: {e}")
                    continue
    finally:
        doc.close()

    print(f'Found {len(all_text)} text blocks and {len(image_paths)} images.')
    return all_text, image_paths
