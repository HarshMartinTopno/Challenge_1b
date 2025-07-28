import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from typing import List, Set, Dict
from app.schemas import Section, SubSection
from collections import Counter
import math

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class HybridPersonaRanker:
    def __init__(self, model_cache: str = None):
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=model_cache)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        
    def extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        # Remove common stop words and short words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter out very common words
        meaningful_words = [w for w in words if len(w) > 3]
        return set(meaningful_words)
    
    def compute_keyword_similarity(self, query: str, text: str) -> float:
        """Compute keyword-based similarity score."""
        query_words = self.extract_keywords(query)
        text_words = self.extract_keywords(text)
        
        if not query_words or not text_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        if union == 0:
            return 0.0
        
        jaccard_score = intersection / union
        
        # Boost for exact phrase matches
        query_lower = query.lower()
        text_lower = text.lower()
        phrase_boost = 0.0
        
        # Check for 2-word and 3-word phrase matches
        query_phrases = self.extract_phrases(query_lower, n=2) + self.extract_phrases(query_lower, n=3)
        for phrase in query_phrases:
            if phrase in text_lower:
                phrase_boost += 0.1
        
        return min(1.0, jaccard_score + phrase_boost)
    
    def extract_phrases(self, text: str, n: int = 2) -> List[str]:
        """Extract n-gram phrases from text."""
        words = text.split()
        phrases = []
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i + n])
            phrases.append(phrase)
        return phrases
    
    def compute_position_score(self, section: Section, total_sections: int) -> float:
        """Compute position-based relevance (earlier pages often more important)."""
        # Normalize by total pages, with diminishing returns
        position_factor = 1.0 / math.sqrt(section.page)
        return min(1.0, position_factor)
    
    def compute_length_score(self, text: str) -> float:
        """Normalize score by text length to avoid bias toward longer sections."""
        word_count = len(text.split())
        # Optimal length around 200-500 words gets highest score
        if 200 <= word_count <= 500:
            return 1.0
        elif word_count < 200:
            return word_count / 200.0
        else:
            return 500.0 / word_count
    
    def rank_sections_hybrid(self, persona_query: str, sections: List[Section], top_k: int = 20) -> List[Section]:
        """Enhanced ranking with hybrid scoring."""
        if not sections:
            return []

        # Batch encode all sections for efficiency
        sec_texts = [s.text if s.text else "" for s in sections]
        sec_emb = self.model.encode(
            sec_texts, 
            convert_to_numpy=True, 
            batch_size=32,
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        
        query_emb = self.model.encode(
            [persona_query], 
            convert_to_numpy=True,
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        
        # Compute semantic similarity
        semantic_scores = cosine_similarity(query_emb, sec_emb)[0]
        
     
        total_sections = len(sections)
        for i, section in enumerate(sections):
      
            semantic_score = float(semantic_scores[i])
       
            keyword_score = self.compute_keyword_similarity(persona_query, section.text)
      
            position_score = self.compute_position_score(section, total_sections)
  
            length_score = self.compute_length_score(section.text)
            
            # Weighted combination
            final_score = (
                0.65 * semantic_score +      # Primary: semantic understanding  
                0.20 * keyword_score +       # Secondary: exact keyword matches
                0.10 * position_score +      # Tertiary: document position
                0.05 * length_score          # Normalization: optimal length
            )
            
            section.score = final_score
        
        # Sort by combined score
        sections.sort(key=lambda s: s.score, reverse=True)
        return sections[:top_k]
    
    def rank_sections(self, persona_query: str, sections: List[Section], top_k: int = 20) -> List[Section]:
        """Main ranking method - uses hybrid approach."""
        return self.rank_sections_hybrid(persona_query, sections, top_k)

    def rank_subsections(self, persona_query: str, section: Section, top_p: int = 3) -> List[SubSection]:
        """Enhanced subsection ranking with hybrid scoring."""
        # Split into paragraphs
        paras = [p.strip() for p in re.split(r"\n{2,}", section.text) if p.strip()]
        if not paras:
            paras = [section.text]

        if len(paras) <= top_p:
          
            return [
                SubSection(
                    document=section.document,
                    page=section.page,
                    parent_section_title=section.section_title,
                    refined_text=para,
                    score=1.0
                )
                for para in paras
            ]

        # Encode paragraphs
        para_emb = self.model.encode(
            paras, 
            convert_to_numpy=True, 
            batch_size=16,
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        query_emb = self.model.encode(
            [persona_query], 
            convert_to_numpy=True, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        
        semantic_scores = cosine_similarity(query_emb, para_emb)[0]
      
        para_scores = []
        for i, para in enumerate(paras):
            semantic_score = float(semantic_scores[i])
            keyword_score = self.compute_keyword_similarity(persona_query, para)
            
    
            combined_score = 0.8 * semantic_score + 0.2 * keyword_score
            para_scores.append((i, combined_score))

        para_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in para_scores[:top_p]]
        
        return [
            SubSection(
                document=section.document,
                page=section.page,
                parent_section_title=section.section_title,
                refined_text=paras[i],
                score=para_scores[top_indices.index(i)][1]
            )
            for i in top_indices
        ]

# Alias for backward compatibility
PersonaRanker = HybridPersonaRanker