"""
Paper Recommender

This module provides vector-based paper recommendation using TF-IDF
embeddings and cosine similarity. It can find related papers based on
content similarity and various filtering criteria.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Set
import logging
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperRecommender:
    """Recommends related papers using vector embeddings and similarity matching"""
    
    def __init__(self, min_similarity: float = 0.1, max_features: int = 5000):
        """
        Initialize the recommender
        
        Args:
            min_similarity: Minimum similarity threshold for recommendations
            max_features: Maximum number of TF-IDF features
        """
        self.min_similarity = min_similarity
        self.max_features = max_features
        
        # Initialize TF-IDF vectorizer with research paper optimizations
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Ignore terms that appear in fewer than 2 documents
            max_df=0.8,  # Ignore terms that appear in more than 80% of documents
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # Only alphabetic tokens
            sublinear_tf=True  # Apply sublinear scaling
        )
        
        self.paper_vectors = None
        self.papers_db = []
        self.paper_index = {}
        self.category_index = defaultdict(list)
        self.author_index = defaultdict(list)
        
    def build_corpus(self, papers: List[Dict]) -> None:
        """
        Build vector corpus from papers
        
        Args:
            papers: List of paper dictionaries
        """
        if not papers:
            logger.warning("No papers provided to build corpus")
            return
        
        logger.info(f"Building corpus from {len(papers)} papers...")
        
        self.papers_db = papers
        self.paper_index = {}
        self.category_index.clear()
        self.author_index.clear()
        
        # Prepare documents and build indices
        documents = []
        for i, paper in enumerate(papers):
            # Combine title, summary, and categories for vectorization
            doc_text = self._prepare_document_text(paper)
            documents.append(doc_text)
            
            # Build paper index
            self.paper_index[paper.get('title', f'Paper_{i}')] = i
            
            # Build category index
            categories = paper.get('categories', [])
            for category in categories:
                self.category_index[category].append(i)
            
            # Build author index
            authors = paper.get('authors', [])
            for author in authors:
                self.author_index[author.lower()].append(i)
        
        # Fit TF-IDF vectorizer and transform documents
        try:
            self.paper_vectors = self.vectorizer.fit_transform(documents)
            logger.info(f"Successfully built corpus with {self.paper_vectors.shape[1]} features")
        except Exception as e:
            logger.error(f"Failed to build corpus: {e}")
            self.paper_vectors = None
    
    def find_similar_papers(self, query_paper: Dict, top_k: int = 5, 
                          filter_categories: Optional[List[str]] = None,
                          exclude_same_authors: bool = False) -> List[Tuple[Dict, float]]:
        """
        Find papers similar to the query paper
        
        Args:
            query_paper: Query paper dictionary
            top_k: Number of recommendations to return
            filter_categories: Only return papers from these categories
            exclude_same_authors: Exclude papers with overlapping authors
            
        Returns:
            List of (paper, similarity_score) tuples
        """
        if self.paper_vectors is None or not self.papers_db:
            logger.warning("Corpus not built. Call build_corpus() first.")
            return []
        
        # Prepare query document
        query_doc = self._prepare_document_text(query_paper)
        
        try:
            # Transform query to vector space
            query_vector = self.vectorizer.transform([query_doc])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.paper_vectors).flatten()
            
            # Get candidate indices sorted by similarity
            candidate_indices = np.argsort(similarities)[::-1]
            
            # Apply filters and collect recommendations
            recommendations = []
            query_authors = set(author.lower() for author in query_paper.get('authors', []))
            query_title = query_paper.get('title', '')
            
            for idx in candidate_indices:
                if len(recommendations) >= top_k:
                    break
                
                similarity = similarities[idx]
                if similarity < self.min_similarity:
                    continue
                
                paper = self.papers_db[idx]
                
                # Skip if same paper (exact title match)
                if paper.get('title', '') == query_title:
                    continue
                
                # Apply category filter
                if filter_categories:
                    paper_categories = paper.get('categories', [])
                    if not any(cat in paper_categories for cat in filter_categories):
                        continue
                
                # Apply author filter
                if exclude_same_authors:
                    paper_authors = set(author.lower() for author in paper.get('authors', []))
                    if query_authors.intersection(paper_authors):
                        continue
                
                recommendations.append((paper, similarity))
            
            logger.info(f"Found {len(recommendations)} similar papers")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to find similar papers: {e}")
            return []
    
    def recommend_by_keywords(self, keywords: List[str], top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Recommend papers based on keyword similarity
        
        Args:
            keywords: List of keywords
            top_k: Number of recommendations
            
        Returns:
            List of (paper, similarity_score) tuples
        """
        if self.paper_vectors is None:
            return []
        
        # Create pseudo-paper from keywords
        pseudo_paper = {
            'title': ' '.join(keywords),
            'summary': ' '.join(keywords),
            'categories': [],
            'authors': []
        }
        
        return self.find_similar_papers(pseudo_paper, top_k)
    
    def recommend_by_category(self, categories: List[str], top_k: int = 5) -> List[Dict]:
        """
        Recommend popular papers from specific categories
        
        Args:
            categories: List of categories to search
            top_k: Number of recommendations
            
        Returns:
            List of paper dictionaries
        """
        candidate_indices = set()
        
        # Collect papers from requested categories
        for category in categories:
            candidate_indices.update(self.category_index.get(category, []))
        
        # Sort by publication date (recent first) and return top-k
        candidates = [(self.papers_db[i], i) for i in candidate_indices]
        candidates.sort(key=lambda x: x[0].get('published', ''), reverse=True)
        
        return [paper for paper, _ in candidates[:top_k]]
    
    def recommend_by_author(self, author_name: str, top_k: int = 5) -> List[Dict]:
        """
        Recommend papers by specific author
        
        Args:
            author_name: Author name to search
            top_k: Number of recommendations
            
        Returns:
            List of paper dictionaries
        """
        author_papers = self.author_index.get(author_name.lower(), [])
        
        # Sort by publication date and return top-k
        papers = [self.papers_db[i] for i in author_papers]
        papers.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        return papers[:top_k]
    
    def get_trending_topics(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """
        Get trending topics based on paper categories
        
        Args:
            top_k: Number of trending topics
            
        Returns:
            List of (category, count) tuples
        """
        category_counts = [(cat, len(papers)) for cat, papers in self.category_index.items()]
        category_counts.sort(key=lambda x: x[1], reverse=True)
        
        return category_counts[:top_k]
    
    def get_paper_clusters(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """
        Cluster papers using K-means on TF-IDF vectors
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster IDs to paper indices
        """
        if self.paper_vectors is None:
            return {}
        
        try:
            from sklearn.cluster import KMeans
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.paper_vectors.toarray())
            
            # Group papers by cluster
            clusters = defaultdict(list)
            for i, cluster_id in enumerate(cluster_labels):
                clusters[cluster_id].append(i)
            
            return dict(clusters)
            
        except ImportError:
            logger.warning("scikit-learn not available for clustering")
            return {}
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}
    
    def _prepare_document_text(self, paper: Dict) -> str:
        """
        Prepare document text for vectorization
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Processed document text
        """
        components = []
        
        # Add title (with higher weight by repeating)
        title = paper.get('title', '')
        if title:
            components.extend([title] * 2)  # Repeat title for emphasis
        
        # Add summary/abstract
        summary = paper.get('summary', '')
        if summary:
            # Clean summary text
            summary = self._clean_text(summary)
            components.append(summary)
        
        # Add categories
        categories = paper.get('categories', [])
        if categories:
            # Repeat categories for emphasis and expand abbreviations
            expanded_categories = []
            for cat in categories:
                expanded_categories.append(cat)
                expanded_categories.append(self._expand_category(cat))
            components.extend(expanded_categories * 2)
        
        # Add primary category with extra weight
        primary_cat = paper.get('primary_category', '')
        if primary_cat:
            components.extend([self._expand_category(primary_cat)] * 3)
        
        return ' '.join(components)
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and newlines
        text = ' '.join(text.split())
        
        # Remove citation markers
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d{4}\)', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters but keep letters, numbers, spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.lower()
    
    def _expand_category(self, category: str) -> str:
        """Expand category abbreviations to full terms"""
        category_map = {
            'cs.AI': 'artificial intelligence computer science',
            'cs.LG': 'machine learning computer science',
            'cs.CV': 'computer vision',
            'cs.CL': 'computational linguistics natural language processing',
            'cs.NE': 'neural networks',
            'cs.RO': 'robotics',
            'stat.ML': 'machine learning statistics',
            'physics.comp-ph': 'computational physics',
            'q-bio.QM': 'quantitative methods biology',
            'math.ST': 'statistics mathematics',
            'econ.EM': 'econometrics economics'
        }
        
        return category_map.get(category, category.replace('.', ' ').replace('-', ' '))
    
    def get_similarity_matrix(self) -> Optional[np.ndarray]:
        """
        Get full similarity matrix between all papers
        
        Returns:
            Similarity matrix or None if corpus not built
        """
        if self.paper_vectors is None:
            return None
        
        return cosine_similarity(self.paper_vectors)
    
    def save_corpus(self, filepath: str) -> None:
        """Save the built corpus to disk"""
        try:
            import pickle
            
            corpus_data = {
                'vectorizer': self.vectorizer,
                'paper_vectors': self.paper_vectors,
                'papers_db': self.papers_db,
                'paper_index': self.paper_index,
                'category_index': dict(self.category_index),
                'author_index': dict(self.author_index)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(corpus_data, f)
            
            logger.info(f"Corpus saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save corpus: {e}")
    
    def load_corpus(self, filepath: str) -> None:
        """Load a previously saved corpus"""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                corpus_data = pickle.load(f)
            
            self.vectorizer = corpus_data['vectorizer']
            self.paper_vectors = corpus_data['paper_vectors']
            self.papers_db = corpus_data['papers_db']
            self.paper_index = corpus_data['paper_index']
            self.category_index = defaultdict(list, corpus_data['category_index'])
            self.author_index = defaultdict(list, corpus_data['author_index'])
            
            logger.info(f"Corpus loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")