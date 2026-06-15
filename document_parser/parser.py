import pdfplumber
from pdf2image import convert_from_path
import os

def extract_text_with_positions(pdf_path):

    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words()

            page_info = {
                "page_number": page_num + 1,
                "page_width": page.width,
                "page_height": page.height,
                "words": words,
                "full_text": page.extract_text()
            }
            pages_data.append(page_info)

    return pages_data

def convert_pages_to_images(pdf_path, output_dir = "data/page_images", dpi=200):

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=dpi)

    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_dir, f"{base_name}_page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return image_paths

if __name__ == "__main__":

    #quick test
    test_pdf = "data/samples/bank_statement_clean_word.pdf"

    print ("Extracting text with positions...")
    pages_data = extract_text_with_positions(test_pdf)
    for page in pages_data:
        print(f"Page {page['page_number']} - Width: {page['page_width']}, Height: {page['page_height']}")
        print(f"Number of words: {len(page['words'])}")
        print(f"First 200 chars of text: {page['full_text'][:200]}")

    print("\nConverting pages to images...")
    image_paths = convert_pages_to_images(test_pdf)
    for path in image_paths:
        print(f"Saved image: {path}")