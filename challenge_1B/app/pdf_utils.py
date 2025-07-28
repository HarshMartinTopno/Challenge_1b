import fitz  # PyMuPDF
import re
import os
from typing import List, Dict, Tuple
from app.schemas import Section

def analyze_font_properties(page) -> Dict[str, any]:
    """Analyze font properties across the page to identify heading patterns."""
    font_stats = {}
    text_blocks = page.get_text("dict")["blocks"]
    
    for block in text_blocks:
        if "lines" not in block:
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                font_size = span["size"]
                font_flags = span["flags"]  # Bold, italic, etc.
                font_name = span["font"]
                
                key = f"{font_name}_{font_size}_{font_flags}"
                if key not in font_stats:
                    font_stats[key] = {"count": 0, "total_chars": 0}
                
                font_stats[key]["count"] += 1
                font_stats[key]["total_chars"] += len(span["text"])
    
    return font_stats

def identify_heading_fonts(font_stats: Dict) -> List[str]:
    """Identify which font combinations are likely headings."""
    if not font_stats:
        return []
    
    # Sort by font size (larger fonts more likely to be headings)
    font_items = [(k, v) for k, v in font_stats.items()]
    font_items.sort(key=lambda x: float(x[0].split('_')[1]), reverse=True)
    
    heading_fonts = []
    total_chars = sum(v["total_chars"] for v in font_stats.values())
    
    for font_key, stats in font_items:
        font_name, font_size, font_flags = font_key.split('_')
        font_size = float(font_size)
        char_ratio = stats["total_chars"] / total_chars if total_chars > 0 else 0
        
        # Heuristics for heading identification:
        # 1. Large font size relative to others
        # 2. Low character ratio (headings are typically short)
        # 3. Bold text (font_flags & 16 indicates bold)
        is_large = font_size > 12
        is_sparse = char_ratio < 0.1  # Less than 10% of total text
        is_bold = int(font_flags) & 16  # Bold flag
        
        if (is_large and is_sparse) or is_bold:
            heading_fonts.append(font_key)
    
    return heading_fonts[:3]  # Top 3 heading font patterns

def extract_sections_with_font_analysis(pdf_path: str) -> List[Section]:
    """Enhanced section extraction using font analysis."""
    doc = fitz.open(pdf_path)
    sections: List[Section] = []
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        
        # Analyze font properties for this page
        font_stats = analyze_font_properties(page)
        heading_fonts = identify_heading_fonts(font_stats)
        
        # Extract text with font information
        text_blocks = page.get_text("dict")["blocks"]
        page_text = ""
        potential_headings = []
        
        for block in text_blocks:
            if "lines" not in block:
                continue
                
            block_text = ""
            for line in block["lines"]:
                line_text = ""
                line_fonts = []
                
                for span in line["spans"]:
                    span_text = span["text"]
                    font_key = f"{span['font']}_{span['size']}_{span['flags']}"
                    line_text += span_text
                    line_fonts.append(font_key)
                
                # Check if this line is likely a heading
                is_heading_line = any(font in heading_fonts for font in line_fonts)
                clean_text = line_text.strip()
                
                if (is_heading_line and clean_text and 
                    len(clean_text) < 150 and len(clean_text.split()) < 15):
                    potential_headings.append(clean_text)
                
                block_text += line_text + "\n"
            
            page_text += block_text + "\n"
        
        # Choose the best heading for this page
        if potential_headings:
            # Prefer shorter, more title-like headings
            section_title = min(potential_headings, key=lambda x: len(x))
        else:
            # Fallback to first non-empty line
            lines = [l.strip() for l in page_text.split('\n') if l.strip()]
            section_title = lines[0] if lines else f"Page {page_index + 1}"
        
        sections.append(Section(
            document=os.path.basename(pdf_path),
            page=page_index + 1,
            section_title=section_title,
            text=page_text.strip()
        ))
    
    doc.close()
    return sections

# Keep the original function as fallback
def extract_sections(pdf_path: str) -> List[Section]:
    """Main extraction function with font analysis."""
    try:
        return extract_sections_with_font_analysis(pdf_path)
    except Exception as e:
        print(f"Font analysis failed for {pdf_path}, falling back to basic extraction: {e}")
        # Fallback to basic extraction (your original method)
        return extract_sections_basic(pdf_path)

def extract_sections_basic(pdf_path: str) -> List[Section]:
    """Basic section extraction (original method as fallback)."""
    doc = fitz.open(pdf_path)
    sections: List[Section] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
        text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        if lines:
            candidate = lines[0]
            is_heading = (
                len(candidate) < 120 and (
                    candidate.isupper() or
                    bool(re.match(r"^([A-Z][a-z]+)(\s[A-Z][a-z]+)*$", candidate)) or
                    len(candidate.split()) < 10
                )
            )
            section_title = candidate if is_heading else f"Page {page_index+1}"
        else:
            section_title = f"Page {page_index+1}"

        sections.append(Section(
            document=os.path.basename(pdf_path),
            page=page_index + 1,
            section_title=section_title,
            text=text
        ))

    doc.close()
    return sections