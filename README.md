# 📄 Research Paper Summarizer & Recommender

A GenAI-powered application that fetches research papers from Arxiv, generates AI summaries, and recommends related papers using vector similarity.

## ✨ Features

- 🔍 **Arxiv Integration**: Search and fetch latest research papers
- 🤖 **AI Summarization**: Generate 3-4 bullet point summaries using OpenAI GPT
- 🔗 **Smart Recommendations**: Find related papers using TF-IDF embeddings
- 🎨 **Interactive UI**: Clean Streamlit interface
- 📊 **Vector Search**: Cosine similarity-based paper matching

## 🚀 Demo

![Demo Screenshot](demo_screenshot.png)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for better summaries)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/research-paper-summarizer.git
cd research-paper-summarizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables (optional)**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

4. **Run the application**
```bash
streamlit run main.py
```

## 📁 Project Structure

```
research-paper-summarizer/
├── main.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore file
├── demo_screenshot.png    # Demo screenshot
└── utils/
    ├── __init__.py
    ├── arxiv_fetcher.py   # Arxiv API integration
    ├── summarizer.py      # AI summarization logic
    └── recommender.py     # Vector-based recommendations
```

## 🔧 Usage

### Basic Usage

1. **Search Papers**: Enter keywords like "machine learning", "neural networks"
2. **View Summaries**: Get AI-generated bullet point summaries
3. **Find Related**: Discover similar papers based on content similarity

### Advanced Configuration

- **OpenAI Integration**: Add your API key for better summaries
- **Search Parameters**: Adjust number of papers to fetch
- **Similarity Threshold**: Modify recommendation sensitivity

### Example Queries

- `machine learning transformers`
- `computer vision deep learning`
- `natural language processing BERT`
- `quantum computing algorithms`

## 🏗️ Architecture

### Components

1. **ArxivPaperFetcher**
   - Interfaces with Arxiv API
   - Parses XML responses
   - Extracts paper metadata

2. **PaperSummarizer**
   - OpenAI GPT integration
   - Extractive summarization fallback
   - Generates 3-4 bullet points

3. **PaperRecommender**
   - TF-IDF vectorization
   - Cosine similarity matching
   - Related paper discovery

### Data Flow

```
Search Query → Arxiv API → Paper Fetching → AI Summarization → Vector Embedding → Similarity Search → Recommendations
```

## 🚦 API Limits & Considerations

- **Arxiv API**: Rate limited to 3 requests per second
- **OpenAI API**: Requires API key and billing setup
- **Fallback**: Uses extractive summarization when OpenAI unavailable

## 🔑 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🧪 Testing

Run the application locally:

```bash
streamlit run main.py
```

Test with different search queries to verify:
- Paper fetching from Arxiv
- Summary generation
- Related paper recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] Support for multiple paper databases (PubMed, IEEE)
- [ ] Advanced filtering (date range, author, journal)
- [ ] Paper bookmark and export features
- [ ] Batch processing capabilities
- [ ] Custom embedding models
- [ ] Paper citation network visualization

## 🐛 Known Issues

- Large abstracts may exceed OpenAI token limits
- XML parsing may fail with malformed Arxiv responses
- Similarity matching depends on corpus size

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Arxiv](https://arxiv.org/) for providing free access to research papers
- [OpenAI](https://openai.com/) for GPT API
- [Streamlit](https://streamlit.io/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for ML utilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/research-paper-summarizer/issues) page
2. Create a new issue with detailed description
3. Contact: your.email@example.com

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/research-paper-summarizer&type=Date)](https://star-history.com/#yourusername/research-paper-summarizer&Date)

---

**Built with ❤️ for the research community**