import argparse
import os
import time
from typing import List
from app.utils import read_json, write_json
from app.schemas import Section, SubSection, now_iso
from app.pdf_utils import extract_sections
from app.ranker import HybridPersonaRanker

def build_persona_query(role: str, task: str) -> str:
    return f"Persona: {role}\nJob-to-be-done: {task}"

def batch_extract_sections(pdf_paths: List[str]) -> List[Section]:
    """Optimized batch extraction of sections from multiple PDFs."""
    all_sections: List[Section] = []
    
    print(f"Extracting sections from {len(pdf_paths)} documents...")
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdf_paths, 1):
        try:
            sections = extract_sections(pdf_path)
            all_sections.extend(sections)
            print(f"  [{i}/{len(pdf_paths)}] Extracted {len(sections)} sections from {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"  [ERROR] Failed to extract from {os.path.basename(pdf_path)}: {e}")
            continue
    
    extraction_time = time.time() - start_time
    print(f"Section extraction completed in {extraction_time:.2f}s. Total sections: {len(all_sections)}")
    
    return all_sections

def main():
    parser = argparse.ArgumentParser(description="Challenge 1B: Persona-Driven Document Intelligence")
    parser.add_argument("--input_dir", default="./input", help="Directory containing input files")
    parser.add_argument("--output_dir", default="./output", help="Directory for output files")
    parser.add_argument("--persona_file", default="./input/persona_job.json", help="Persona configuration file")
    parser.add_argument("--top_sections", type=int, default=15, help="Number of top sections to extract")
    parser.add_argument("--top_subsections", type=int, default=3, help="Number of top subsections per section")
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load and parse configuration ----
    print("Loading configuration...")
    try:
        raw = read_json(args.persona_file)
        persona_role = raw["persona"]["role"]
        job_task = raw["job_to_be_done"]["task"]
        input_documents: List[str] = [d["filename"] for d in raw["documents"]]
        
        print(f"Persona: {persona_role}")
        print(f"Job: {job_task}")
        print(f"Documents: {len(input_documents)}")
        
    except Exception as e:
        raise RuntimeError(f"Invalid persona_job.json structure: {e}")

    # ---- Validate PDF files ----
    pdf_paths = []
    missing_files = []
    
    for fname in input_documents:
        full_path = os.path.join(args.input_dir, fname)
        if os.path.exists(full_path):
            pdf_paths.append(full_path)
        else:
            missing_files.append(fname)
    
    if missing_files:
        print(f"WARNING: Missing files: {missing_files}")
    
    if not pdf_paths:
        raise FileNotFoundError("No valid PDF files found")

    persona_query = build_persona_query(persona_role, job_task)

    # ---- Batch extract sections ----
    all_sections = batch_extract_sections(pdf_paths)
    
    if not all_sections:
        raise RuntimeError("No sections extracted from any document")

    # ---- Hybrid ranking ----
    print(f"Ranking sections using hybrid approach...")
    ranking_start = time.time()
    
    ranker = HybridPersonaRanker()
    top_sections = ranker.rank_sections(persona_query, all_sections, top_k=args.top_sections)
    
    ranking_time = time.time() - ranking_start
    print(f"Ranking completed in {ranking_time:.2f}s. Selected {len(top_sections)} top sections.")

    # ---- Build output with enhanced subsection analysis ----
    print("Generating refined summaries...")
    
    extracted_sections = []
    subsection_analysis = []

    for i, section in enumerate(top_sections, start=1):
        # Add to extracted sections
        extracted_sections.append({
            "document": section.document,
            "page_number": section.page,
            "section_title": section.section_title,
            "importance_rank": i
        })
        
        # Get top subsections for this section
        subsections = ranker.rank_subsections(
            persona_query, 
            section, 
            top_p=args.top_subsections
        )
        
        # Add subsections to analysis
        for subsection in subsections:
            subsection_analysis.append({
                "document": subsection.document,
                "refined_text": subsection.refined_text,
                "page_number": subsection.page
            })

    # ---- Prepare output ----
    output = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona_role,
            "job_to_be_done": job_task,
            "processing_timestamp": now_iso(),
            "processing_stats": {
                "total_sections_extracted": len(all_sections),
                "top_sections_selected": len(top_sections),
                "total_processing_time_seconds": round(time.time() - start_time, 2)
            }
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    # ---- Write output ----
    out_file = os.path.join(args.output_dir, "output.json")
    write_json(output, out_file)
    
    total_time = time.time() - start_time
    print(f"\n=== Processing Complete ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Results saved to: {out_file}")
    print(f"Sections processed: {len(all_sections)}")
    print(f"Top sections selected: {len(top_sections)}")
    print(f"Subsections analyzed: {len(subsection_analysis)}")

if __name__ == "__main__":
    main()