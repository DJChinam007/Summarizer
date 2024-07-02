import os
from unstructured.partition.pdf import partition_pdf
from latex2mathml.converter import convert as latex_to_mathml
import re
def extract_pdf_elements(filename, output_dir):
    return partition_pdf(
        filename=filename,
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=output_dir
    )

def categorize_elements(elements):
    categories = {
        "Header": [], "Footer": [], "Title": [], "NarrativeText": [],
        "Text": [], "ListItem": [], "Images": [], "Tables": []
    }

    for element in elements:
        element_type = type(element).__name__
        if element_type in categories:
            categories[element_type].append(str(element))
        elif element_type == "Image":
            categories["Images"].append(str(element))
        elif element_type == "Table":
            categories["Tables"].append(str(element))

    return categories

def process_pdf(filename, temp_dir):
    extract_dir = os.path.join(temp_dir, "extracted_data")
    os.makedirs(extract_dir, exist_ok=True)

    raw_pdf_elements = extract_pdf_elements(filename, extract_dir)
    categorized_elements = categorize_elements(raw_pdf_elements)

    categorized_elements["Images"] = [
        os.path.join(extract_dir, img) for img in os.listdir(extract_dir) if img.endswith(('.png', '.jpg', '.jpeg'))
    ]

    return categorized_elements

def clean_table_data(table_string):
    # Split the table into rows
    rows = table_string.strip().split('\n')
    
    # Process each row
    processed_rows = []
    for row in rows:
        # Replace multiple spaces with a single space
        row = re.sub(r'\s+', ' ', row.strip())
        # Split the row into columns
        columns = row.split(' ')
        processed_rows.append(columns)
    
    # Find the maximum number of columns
    max_columns = max(len(row) for row in processed_rows)
    
    # Pad rows with fewer columns
    for row in processed_rows:
        row.extend([''] * (max_columns - len(row)))
    
    return processed_rows

def process_latex(text):
    # Find LaTeX expressions enclosed in $ or $$
    latex_pattern = r'\$\$(.*?)\$\$|\$(.*?)\$'
    
    def replace_latex(match):
        latex = match.group(1) or match.group(2)
        return latex_to_mathml(latex)
    
    return re.sub(latex_pattern, replace_latex, text)
