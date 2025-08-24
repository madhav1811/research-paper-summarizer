"""
Alternative Streamlit App Structure

This is an alternative main app file that imports utilities
from the utils package for cleaner code organization.
"""

import streamlit as st
import os
from dotenv import load_dotenv
from utils import ArxivPaperFetcher, PaperSummarizer, PaperRecommender
import time

# Load environment variables
load_dotenv()

def init_session_state():
    """Initialize session state variables"""
    if 'papers' not in st.session_state:
        st.session_state.papers = []
    if 'recommender_built' not in st.session_state:
        st.session_state.recommender_built = False
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def display_paper_card(paper, idx, summarizer, recommender):
    """Display a paper in an attractive card format"""
    with st.container():
        # Paper header
        st.markdown(f"### 📄 {paper['title']}")
        
        # Metadata row
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            authors_text = ", ".join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors_text += f" +{len(paper['authors']) - 3} more"
            st.markdown(f"**Authors:** {authors_text}")
        
        with col2:
            st.markdown(f"**Date:** {paper['published'][:10]}")
        
        with col3:
            if paper.get('pdf_link'):
                st.link_button("📑 PDF", paper['pdf_link'])
        
        # Categories
        if paper.get('categories'):
            categories_str = " • ".join(paper['categories'][:4])
            st.markdown(f"**Categories:** {categories_str}")
        
        # Tabbed content
        tab1, tab2, tab3 = st.tabs(["🤖 AI Summary", "📋 Abstract", "🔗 Related"])
        
        with tab1:
            with st.spinner("Generating AI summary..."):
                summary = summarizer.summarize_paper(paper)
            
            for i, point in enumerate(summary, 1):
                st.markdown(f"**{i}.** {point}")
        
        with tab2:
            st.write(paper['summary'])
        
        with tab3:
            if st.session_state.recommender_built:
                similar_papers = recommender.find_similar_papers(paper, top_k=3)
                
                if similar_papers:
                    for related_paper, similarity in similar_papers:
                        with st.expander(f"{related_paper['title'][:60]}... (Score: {similarity:.3f})"):
                            st.markdown(f"**Authors:** {', '.join(related_paper['authors'][:2])}")
                            st.markdown(f"**Date:** {related_paper['published'][:10]}")
                            st.write(related_paper['summary'][:300] + "...")
                else:
                    st.info("No similar papers found in current results")
            else:
                st.info("Search more papers to enable recommendations")
        
        st.divider()

def main():
    st.set_page_config(
        page_title="Research Paper AI Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🔬 Research Paper AI Assistant")
    st.markdown("*AI-powered paper discovery, summarization & recommendations*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Configuration
        openai_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Optional: For enhanced AI summaries"
        )
        
        # Search Parameters
        st.subheader("🔍 Search Settings")
        max_results = st.slider("Papers to fetch", 5, 50, 15)
        
        # Advanced filters
        with st.expander("🎯 Advanced Filters"):
            category_filter = st.selectbox(
                "Category Filter",
                ["None", "cs.AI", "cs.LG", "cs.CV", "cs.CL", "stat.ML"],
                help="Filter by Arxiv category"
            )
            
            date_filter = st.selectbox(
                "Date Range",
                ["Any time", "Last month", "Last 6 months", "Last year"]
            )
        
        # Search History
        if st.session_state.search_history:
            st.subheader("📚 Recent Searches")
            for i, query in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"🔄 {query}", key=f"hist_{i}"):
                    st.session_state.current_query = query
                    st.rerun()
        
        # Statistics
        if st.session_state.papers:
            st.subheader("📊 Statistics")
            st.metric("Papers Loaded", len(st.session_state.papers))
            
            # Category distribution
            categories = {}
            for paper in st.session_state.papers:
                for cat in paper.get('categories', []):
                    categories[cat] = categories.get(cat, 0) + 1
            
            if categories:
                top_category = max(categories.items(), key=lambda x: x[1])
                st.metric("Top Category", top_category[0], top_category[1])
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "🔍 Search Research Papers",
            placeholder="e.g., transformer neural networks, quantum computing, machine learning",
            key="search_input"
        )
    
    with col2:
        search_pressed = st.button("🚀 Search", type="primary", use_container_width=True)
    
    # Quick search suggestions
    st.markdown("**💡 Suggestions:** *transformer attention*, *computer vision*, *reinforcement learning*, *quantum algorithms*")
    
    # Initialize components
    fetcher = ArxivPaperFetcher()
    summarizer = PaperSummarizer(openai_key if openai_key else None)
    recommender = PaperRecommender()
    
    # Handle search
    if search_pressed or (hasattr(st.session_state, 'current_query') and st.session_state.current_query):
        query = getattr(st.session_state, 'current_query', search_query)
        if hasattr(st.session_state, 'current_query'):
            delattr(st.session_state, 'current_query')
        
        if query:
            # Add to search history
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch papers
            status_text.text("🔍 Searching Arxiv database...")
            progress_bar.progress(20)
            
            # Apply category filter to query
            if category_filter != "None":
                query = f"cat:{category_filter} AND {query}"
            
            papers = fetcher.search_papers(query, max_results=max_results)
            
            if papers:
                progress_bar.progress(50)
                status_text.text("🧠 Building recommendation system...")
                
                # Build recommender corpus
                recommender.build_corpus(papers)
                st.session_state.recommender_built = True
                
                progress_bar.progress(80)
                status_text.text("✅ Processing complete!")
                
                # Store in session state
                st.session_state.papers = papers
                
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Success message
                st.success(f"🎉 Found {len(papers)} papers! Scroll down to explore.")
                
            else:
                st.error("❌ No papers found. Try different keywords or check your connection.")
                st.stop()
    
    # Display results
    if st.session_state.papers:
        # Results header
        st.header(f"📚 Research Papers ({len(st.session_state.papers)} found)")
        
        # Sorting options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_by = st.selectbox(
                "📈 Sort by:",
                ["Relevance", "Date (newest)", "Date (oldest)", "Author"]
            )
        
        with col2:
            view_mode = st.selectbox("👁️ View:", ["Detailed", "Compact"])
        
        with col3:
            papers_per_page = st.selectbox("📄 Per page:", [5, 10, 15, 20])
        
        # Sort papers
        if sort_by == "Date (newest)":
            papers = sorted(st.session_state.papers, key=lambda x: x.get('published', ''), reverse=True)
        elif sort_by == "Date (oldest)":
            papers = sorted(st.session_state.papers, key=lambda x: x.get('published', ''))
        elif sort_by == "Author":
            papers = sorted(st.session_state.papers, key=lambda x: x.get('authors', [''])[0])
        else:
            papers = st.session_state.papers  # Keep original relevance order
        
        # Pagination
        total_papers = len(papers)
        total_pages = (total_papers - 1) // papers_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox(f"📖 Page (1-{total_pages}):", range(1, total_pages + 1)) - 1
        else:
            page = 0
        
        start_idx = page * papers_per_page
        end_idx = min(start_idx + papers_per_page, total_papers)
        current_papers = papers[start_idx:end_idx]
        
        # Display papers
        if view_mode == "Detailed":
            for i, paper in enumerate(current_papers):
                display_paper_card(paper, start_idx + i, summarizer, recommender)
        
        else:  # Compact view
            for i, paper in enumerate(current_papers):
                with st.expander(f"📄 {paper['title']}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Authors:** {', '.join(paper['authors'][:2])}{'...' if len(paper['authors']) > 2 else ''}")
                        st.markdown(f"**Published:** {paper['published'][:10]}")
                        st.write(paper['summary'][:200] + "...")
                    
                    with col2:
                        if paper.get('pdf_link'):
                            st.link_button("📑 PDF", paper['pdf_link'])
                        
                        if st.button(f"🤖 Summarize", key=f"sum_{start_idx + i}"):
                            summary = summarizer.summarize_paper(paper)
                            for j, point in enumerate(summary, 1):
                                st.markdown(f"**{j}.** {point}")
        
        # Bulk actions
        if st.session_state.papers:
            st.subheader("🔧 Bulk Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Generate Report"):
                    with st.spinner("Generating research overview..."):
                        # Create a simple research overview
                        categories = {}
                        authors = {}
                        
                        for paper in st.session_state.papers:
                            for cat in paper.get('categories', []):
                                categories[cat] = categories.get(cat, 0) + 1
                            
                            for author in paper.get('authors', [])[:1]:  # First author only
                                authors[author] = authors.get(author, 0) + 1
                        
                        # Display report
                        st.markdown("### 📈 Research Overview")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Top Categories:**")
                            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                                st.markdown(f"• {cat}: {count} papers")
                        
                        with col2:
                            st.markdown("**Frequent Authors:**")
                            for author, count in sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]:
                                if count > 1:
                                    st.markdown(f"• {author}: {count} papers")
            
            with col2:
                if st.button("💾 Export Data"):
                    # Convert to simple format for download
                    export_data = []
                    for paper in st.session_state.papers:
                        export_data.append({
                            "title": paper.get('title', ''),
                            "authors": '; '.join(paper.get('authors', [])),
                            "published": paper.get('published', ''),
                            "categories": '; '.join(paper.get('categories', [])),
                            "pdf_link": paper.get('pdf_link', ''),
                            "summary": paper.get('summary', '')[:500] + "..."
                        })
                    
                    import json
                    st.download_button(
                        "📥 Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name="research_papers.json",
                        mime="application/json"
                    )
            
            with col3:
                if st.button("🎯 Find Trending"):
                    if st.session_state.recommender_built:
                        trending = recommender.get_trending_topics(top_k=8)
                        
                        st.markdown("### 🔥 Trending Topics")
                        for topic, count in trending:
                            st.markdown(f"• **{topic}**: {count} papers")
                    else:
                        st.info("Build corpus first by searching papers")
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### 🔬 Research Paper AI Assistant
        
        **Features:**
        - 🔍 **Smart Search**: Query Arxiv's vast research database
        - 🤖 **AI Summaries**: Get key insights in bullet points
        - 🔗 **Recommendations**: Find related papers using vector similarity
        - 📊 **Analytics**: Discover trending topics and research patterns
        
        **How it works:**
        1. **Search**: Enter keywords to find relevant papers
        2. **Analyze**: AI generates concise summaries
        3. **Explore**: Discover related work through recommendations
        4. **Export**: Save findings for later reference
        
        **Technologies:**
        - Arxiv API for paper retrieval
        - OpenAI GPT for intelligent summarization
        - TF-IDF + Cosine Similarity for recommendations
        - Streamlit for interactive interface
        
        *Built for researchers, by researchers* 🎓
        """)

if __name__ == "__main__":
    main()