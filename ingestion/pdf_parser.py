#PDF->clean text
import fitz  # PyMuPDF

def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        pages.append(text)
    full_text = "\n".join(pages)
    return clean_text(full_text)

def clean_text(text: str) -> str:
    # 简化版，可后续加强
    text = text.replace('\n', ' ')
    return text
