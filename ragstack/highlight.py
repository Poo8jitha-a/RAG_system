from typing import List, Dict, Any, Tuple
import fitz, json, os

def _rects_for_phrase(page, phrase: str):
    rects = []
    for inst in page.search_for(phrase, hit_max=50):
        rects.append(inst)
    return rects

def highlight_pdf(source_pdf: str, page_num: int, phrases: List[str], out_pdf: str) -> str:
    doc = fitz.open(source_pdf)
    page = doc.load_page(page_num - 1)
    for ph in phrases:
        rects = _rects_for_phrase(page, ph)
        for r in rects:
            annot = page.add_highlight_annot(r)
            if annot:
                annot.update()
    doc.save(out_pdf, deflate=True)
    return out_pdf
