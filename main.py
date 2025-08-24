import streamlit as st
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from typing import List, Dict, Tuple
import re
import time

class ArxivPaperFetcher:
    """Fetches papers from Arxiv API"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def search_papers(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search papers on Arxiv"""
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_entry(entry)
                papers.append(paper)
            
            return papers
        
        except requests.RequestException as e:
            st.error(f"Error fetching papers: {e}")
            return []
    
    def _parse_entry(self, entry) -> Dict:
        """Parse individual paper entry from XML"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns).text
            authors.append(name)
        
        # Extract publication date
        published = entry.find('atom:published', ns).text
        
        # Extract PDF link
        pdf_link = None
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'application/pdf':
                pdf_link = link.get('href')
                break
        
        # Extract categories
        categories = []
        for category in entry.findall('atom:category', ns):
            categories.append(category.get('term'))
        
        return {
            'title': title,
            'authors': authors,
            'summary': summary,
            'published': published,
            'pdf_link': pdf_link,
            'categories': categories
        }

class PaperSummarizer:
    """Summarizes papers using various methods"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def summarize_with_openai(self, text: str) -> List[str]:
        """Summarize using OpenAI GPT"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research paper summarizer. Summarize the following paper abstract into exactly 3-4 concise bullet points that capture the key contributions, methodology, and results."},
                    {"role": "user", "content": f"Abstract: {text}"}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            # Split into bullet points
            bullet_points = [point.strip('• -').strip() for point in summary.split('\n') if point.strip()]
            return bullet_points[:4]  # Ensure max 4 points
            
        except Exception as e:
            st.warning(f"OpenAI summarization failed: {e}")
            return self.summarize_extractive(text)
    
    def summarize_extractive(self, text: str) -> List[str]:
        """Simple extractive summarization fallback"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Score sentences by length and position
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split()) * (1 - i/len(sentences) * 0.5)
            scored_sentences.append((score, sentence))
        
        # Get top 3-4 sentences
        scored_sentences.sort(reverse=True)
        summary = [sent[1] for sent in scored_sentences[:4]]
        
        return summary

class PaperRecommender:
    """Recommends related papers using embeddings"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.paper_vectors = None
        self.papers_db = []
    
    def build_corpus(self, papers: List[Dict]):
        """Build vector corpus from papers"""
        self.papers_db = papers
        
        # Combine title and summary for vectorization
        documents = []
        for paper in papers:
            doc = f"{paper['title']} {paper['summary']}"
            documents.append(doc)
        
        if documents:
            self.paper_vectors = self.vectorizer.fit_transform(documents)
    
    def find_similar_papers(self, query_paper: Dict, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Find papers similar to the query paper"""
        if not self.paper_vectors:
            return []
        
        query_doc = f"{query_paper['title']} {query_paper['summary']}"
        query_vector = self.vectorizer.transform([query_doc])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.paper_vectors).flatten()
        
        # Get top-k similar papers (excluding the query paper itself)
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                paper = self.papers_db[idx]
                # Skip if it's the same paper
                if paper['title'] != query_paper['title']:
                    recommendations.append((paper, similarities[idx]))
                
                if len(recommendations) >= top_k:
                    break
        
        return recommendations

def main():
    st.set_page_config(
        page_title="Research Paper Summarizer & Recommender",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Research Paper Summarizer & Recommender")
    st.markdown("*Powered by Arxiv API & AI-based Summarization*")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # OpenAI API key input
    openai_key = st.sidebar.text_input(
        "OpenAI API Key (optional)",
        type="password",
        help="Enter your OpenAI API key for better summarization. Leave empty for basic summarization."
    )
    
    # Search parameters
    max_results = st.sidebar.slider("Max Papers to Fetch", 5, 50, 20)
    
    # Main interface
    st.header("🔍 Search Research Papers")
    
    # Search input
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "Enter search terms (e.g., 'machine learning', 'neural networks', 'quantum computing')",
            placeholder="machine learning transformers"
        )
    
    with col2:
        search_button = st.button("🔍 Search Papers", type="primary")
    
    # Example queries
    st.markdown("**Example queries:** *machine learning*, *computer vision*, *natural language processing*, *quantum computing*")
    
    # Initialize components
    fetcher = ArxivPaperFetcher()
    summarizer = PaperSummarizer(openai_key if openai_key else None)
    recommender = PaperRecommender()
    
    # Search and display results
    if search_button and search_query:
        with st.spinner("🔍 Fetching papers from Arxiv..."):
            papers = fetcher.search_papers(search_query, max_results)
        
        if papers:
            st.success(f"Found {len(papers)} papers!")
            
            # Build recommendation corpus
            with st.spinner("🧠 Building recommendation system..."):
                recommender.build_corpus(papers)
            
            # Display papers
            for i, paper in enumerate(papers):
                with st.expander(f"📄 {paper['title'][:100]}..." if len(paper['title']) > 100 else f"📄 {paper['title']}"):
                    
                    # Paper metadata
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                        st.markdown(f"**Published:** {paper['published'][:10]}")
                        st.markdown(f"**Categories:** {', '.join(paper['categories'][:3])}")
                    
                    with col2:
                        if paper['pdf_link']:
                            st.markdown(f"[📑 View PDF]({paper['pdf_link']})")
                    
                    # Original abstract
                    with st.expander("📋 Original Abstract"):
                        st.write(paper['summary'])
                    
                    # AI Summary
                    st.markdown("### 🤖 AI Summary")
                    with st.spinner("Generating summary..."):
                        summary_points = summarizer.summarize_with_openai(paper['summary']) if openai_key else summarizer.summarize_extractive(paper['summary'])
                    
                    for j, point in enumerate(summary_points, 1):
                        st.markdown(f"**{j}.** {point}")
                    
                    # Related papers
                    st.markdown("### 🔗 Related Papers")
                    similar_papers = recommender.find_similar_papers(paper, top_k=3)
                    
                    if similar_papers:
                        for related_paper, similarity in similar_papers:
                            st.markdown(f"• **[{related_paper['title'][:80]}...]** (Similarity: {similarity:.2f})")
                            st.markdown(f"  *{related_paper['authors'][0]} et al. - {related_paper['published'][:10]}*")
                    else:
                        st.info("No similar papers found in current search results.")
                    
                    st.markdown("---")
        
        else:
            st.warning("No papers found. Try different search terms.")
    
    elif search_button and not search_query:
        st.warning("Please enter search terms.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. 🔍 **Fetch**: Retrieves latest research papers from Arxiv API
    2. 🤖 **Summarize**: Uses AI to generate 3-4 key bullet points
    3. 🔗 **Recommend**: Finds related papers using TF-IDF + cosine similarity
    
    *Built with Streamlit, Arxiv API, and OpenAI*
    """)

if __name__ == "__main__":
    main()