"""
Paper Summarizer

This module provides AI-powered and extractive summarization capabilities
for research papers. It supports OpenAI GPT integration with fallback
to extractive summarization methods.
"""

import openai
import re
import logging
from typing import List, Dict, Optional
from collections import Counter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperSummarizer:
    """Summarizes research papers using various AI and traditional methods"""
    
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the summarizer
        
        Args:
            openai_api_key: OpenAI API key for GPT-based summarization
            model: OpenAI model to use ('gpt-3.5-turbo' or 'gpt-4')
        """
        self.openai_api_key = openai_api_key
        self.model = model
        
        if openai_api_key:
            openai.api_key = openai_api_key
            logger.info(f"Initialized OpenAI summarizer with model: {model}")
        else:
            logger.info("Initialized without OpenAI key - using extractive summarization only")
    
    def summarize_paper(self, paper: Dict, method: str = "auto") -> List[str]:
        """
        Summarize a research paper
        
        Args:
            paper: Paper dictionary with 'title' and 'summary' keys
            method: Summarization method ('openai', 'extractive', 'auto')
            
        Returns:
            List of 3-4 bullet point summaries
        """
        text = paper.get('summary', '')
        title = paper.get('title', '')
        
        if not text:
            return ["No abstract available for summarization."]
        
        if method == "auto":
            method = "openai" if self.openai_api_key else "extractive"
        
        if method == "openai" and self.openai_api_key:
            return self._summarize_with_openai(text, title)
        else:
            return self._summarize_extractive(text, title)
    
    def _summarize_with_openai(self, text: str, title: str = "") -> List[str]:
        """
        Summarize using OpenAI GPT
        
        Args:
            text: Paper abstract text
            title: Paper title for context
            
        Returns:
            List of bullet point summaries
        """
        try:
            # Create prompt with context
            system_prompt = """You are an expert research paper summarizer. Your task is to create exactly 3-4 concise bullet points that capture:
1. The main problem/objective
2. The key methodology or approach  
3. The main findings/results
4. The significance/impact (if space allows)

Each bullet point should be 1-2 sentences and focus on the most important aspects."""

            user_prompt = f"""Title: {title}

Abstract: {text}

Provide exactly 3-4 bullet points summarizing this research paper:"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.3,
                top_p=0.9
            )
            
            summary_text = response.choices[0].message.content.strip()
            
            # Parse bullet points
            bullet_points = self._parse_bullet_points(summary_text)
            
            # Ensure we have 3-4 points
            if len(bullet_points) < 3:
                logger.warning("OpenAI returned fewer than 3 points, falling back to extractive")
                return self._summarize_extractive(text, title)
            
            return bullet_points[:4]  # Ensure max 4 points
            
        except openai.error.RateLimitError:
            logger.warning("OpenAI rate limit exceeded, falling back to extractive summarization")
            return self._summarize_extractive(text, title)
        except openai.error.AuthenticationError:
            logger.error("OpenAI authentication failed, falling back to extractive summarization")
            return self._summarize_extractive(text, title)
        except Exception as e:
            logger.warning(f"OpenAI summarization failed: {e}, falling back to extractive")
            return self._summarize_extractive(text, title)
    
    def _summarize_extractive(self, text: str, title: str = "") -> List[str]:
        """
        Extractive summarization using sentence scoring
        
        Args:
            text: Paper abstract text
            title: Paper title for keyword extraction
            
        Returns:
            List of bullet point summaries
        """
        try:
            # Clean and split text into sentences
            sentences = self._split_sentences(text)
            
            if len(sentences) <= 3:
                return [s.strip() for s in sentences if s.strip()]
            
            # Score sentences based on multiple factors
            scored_sentences = self._score_sentences(sentences, title)
            
            # Select top sentences ensuring diversity
            selected = self._select_diverse_sentences(scored_sentences, target_count=4)
            
            # Format as bullet points
            summary_points = []
            for i, (score, sentence, position) in enumerate(selected):
                # Clean up sentence
                clean_sentence = self._clean_sentence(sentence)
                if clean_sentence:
                    summary_points.append(clean_sentence)
            
            return summary_points[:4]
            
        except Exception as e:
            logger.error(f"Extractive summarization failed: {e}")
            # Return first few sentences as fallback
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() + "." for s in sentences[:3] if len(s.strip()) > 10]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        # Handle common abbreviations
        text = re.sub(r'\b(e\.g\.|i\.e\.|et al\.|vs\.|etc\.)', lambda m: m.group().replace('.', '<!DOT!>'), text)
        
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        # Restore dots and clean up
        sentences = [s.replace('<!DOT!>', '.').strip() for s in sentences]
        sentences = [s for s in sentences if len(s) > 15]  # Filter very short sentences
        
        return sentences
    
    def _score_sentences(self, sentences: List[str], title: str = "") -> List[tuple]:
        """
        Score sentences based on multiple factors
        
        Args:
            sentences: List of sentences to score
            title: Paper title for keyword extraction
            
        Returns:
            List of (score, sentence, position) tuples
        """
        scored = []
        
        # Extract keywords from title
        title_words = set(re.findall(r'\b\w+\b', title.lower())) if title else set()
        
        # Common important research paper keywords
        important_words = {
            'results', 'findings', 'conclusion', 'propose', 'method', 'approach',
            'novel', 'new', 'improved', 'significant', 'performance', 'accuracy',
            'model', 'algorithm', 'framework', 'system', 'evaluation', 'experiment',
            'demonstrate', 'show', 'achieve', 'outperform', 'effective', 'efficient'
        }
        
        for i, sentence in enumerate(sentences):
            score = 0
            words = set(re.findall(r'\b\w+\b', sentence.lower()))
            
            # Position score (earlier sentences often more important)
            position_score = 1.0 - (i / len(sentences)) * 0.3
            score += position_score
            
            # Length score (prefer medium-length sentences)
            length = len(sentence.split())
            if 10 <= length <= 30:
                score += 0.5
            elif length > 30:
                score += 0.2
            
            # Title keyword overlap
            title_overlap = len(words.intersection(title_words))
            score += title_overlap * 0.3
            
            # Important research keywords
            important_overlap = len(words.intersection(important_words))
            score += important_overlap * 0.2
            
            # Avoid sentences that are just citations or references
            if re.search(r'\[\d+\]|\(\d{4}\)|et al\.|Fig\.|Figure', sentence):
                score *= 0.7
            
            # Prefer sentences with numbers/percentages (often results)
            if re.search(r'\d+%|\d+\.\d+|\d+ times', sentence):
                score += 0.3
            
            scored.append((score, sentence, i))
        
        return sorted(scored, key=lambda x: x[0], reverse=True)
    
    def _select_diverse_sentences(self, scored_sentences: List[tuple], 
                                target_count: int = 4) -> List[tuple]:
        """
        Select diverse sentences avoiding redundancy
        
        Args:
            scored_sentences: List of (score, sentence, position) tuples
            target_count: Target number of sentences
            
        Returns:
            Selected sentences
        """
        if len(scored_sentences) <= target_count:
            return scored_sentences
        
        selected = []
        used_words = set()
        
        for score, sentence, position in scored_sentences:
            if len(selected) >= target_count:
                break
            
            # Check for redundancy
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(sentence_words.intersection(used_words))
            
            # Select if not too redundant
            if overlap < len(sentence_words) * 0.6:  # Less than 60% overlap
                selected.append((score, sentence, position))
                used_words.update(sentence_words)
        
        # Sort by original position to maintain flow
        return sorted(selected, key=lambda x: x[2])
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and format sentence"""
        # Remove extra whitespace
        sentence = ' '.join(sentence.split())
        
        # Ensure sentence ends with period
        if sentence and not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        
        return sentence
    
    def _parse_bullet_points(self, text: str) -> List[str]:
        """Parse bullet points from GPT response"""
        lines = text.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove bullet point markers
            line = re.sub(r'^[-•*]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            
            if len(line) > 10:  # Minimum length check
                bullet_points.append(line)
        
        return bullet_points
    
    def batch_summarize(self, papers: List[Dict], method: str = "auto") -> Dict[str, List[str]]:
        """
        Batch summarize multiple papers
        
        Args:
            papers: List of paper dictionaries
            method: Summarization method
            
        Returns:
            Dictionary mapping paper titles to summaries
        """
        summaries = {}
        
        for i, paper in enumerate(papers):
            try:
                logger.info(f"Summarizing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
                summary = self.summarize_paper(paper, method)
                summaries[paper.get('title', f'Paper_{i+1}')] = summary
            except Exception as e:
                logger.error(f"Failed to summarize paper {i+1}: {e}")
                summaries[paper.get('title', f'Paper_{i+1}')] = ["Summarization failed."]
        
        return summaries