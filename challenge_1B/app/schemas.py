from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Section(BaseModel):
    document: str
    page: int
    section_title: str
    text: str
    score: Optional[float] = None

class SubSection(BaseModel):
    document: str
    page: int
    parent_section_title: str
    refined_text: str
    score: Optional[float] = None

def now_iso() -> str:
    return datetime.utcnow().isoformat()