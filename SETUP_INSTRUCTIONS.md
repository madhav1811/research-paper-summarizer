# 🚀 Setup Instructions for Research Paper Summarizer

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Git**: For version control
- **OpenAI API Key**: (Optional) For enhanced AI summaries

## 🔧 Local Development Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/yourusername/research-paper-summarizer.git
cd research-paper-summarizer
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your OpenAI API key (optional)
# OPENAI_API_KEY=your_api_key_here
```

### 5. Run the Application
```bash
# Using main.py (single file version)
streamlit run main.py

# OR using modular version
streamlit run streamlit_app.py
```

### 6. Open in Browser
- Navigate to: `http://localhost:8501`

## 🐳 Docker Setup (Alternative)

### 1. Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Build & Run
```bash
# Build Docker image
docker build -t research-paper-app .

# Run container
docker run -p 8501:8501 research-paper-app
```

## ☁️ Cloud Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets in Streamlit Cloud dashboard:
   - `OPENAI_API_KEY = your_key_here`

### Heroku Deployment
1. Create `Procfile`:
```
web: sh setup.sh && streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## 🧪 Testing

### Run Basic Tests
```bash
# Test Arxiv fetching
python -c "from utils.arxiv_fetcher import ArxivPaperFetcher; f=ArxivPaperFetcher(); print(len(f.search_papers('machine learning', 5)))"

# Test summarization
python -c "from utils.summarizer import PaperSummarizer; s=PaperSummarizer(); print(s.summarize_extractive('This is a test abstract about machine learning.'))"
```

### Interactive Testing
```bash
# Start Python REPL
python

# Test components
from utils import ArxivPaperFetcher, PaperSummarizer, PaperRecommender

# Fetch some papers
fetcher = ArxivPaperFetcher()
papers = fetcher.search_papers("neural networks", 5)
print(f"Found {len(papers)} papers")

# Test summarization
summarizer = PaperSummarizer()
summary = summarizer.summarize_paper(papers[0])
print("Summary:", summary)
```

## 🔧 Configuration Options

### Environment Variables
```bash
# .env file options
OPENAI_API_KEY=sk-your-key-here
DEFAULT_MAX_RESULTS=20
DEFAULT_SIMILARITY_THRESHOLD=0.1
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
maxUploadSize = 200
```

## 📦 Production Deployment Checklist

- [ ] Set up environment variables securely
- [ ] Configure proper logging
- [ ] Set up monitoring (optional)
- [ ] Enable HTTPS
- [ ] Configure caching for better performance
- [ ] Set up backup for any saved data
- [ ] Test with different paper queries
- [ ] Verify OpenAI integration works
- [ ] Check error handling

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the correct directory and virtual environment is activated
pip install -r requirements.txt
```

**2. OpenAI API Issues**
- Check your API key is valid
- Ensure you have sufficient credits
- Try without OpenAI (app will fall back to extractive summarization)

**3. Streamlit Port Issues**
```bash
# Use different port
streamlit run main.py --server.port=8502
```

**4. Memory Issues with Large Paper Sets**
- Reduce `max_results` in the app
- Restart the application periodically

### Getting Help

1. Check the [GitHub Issues](https://github.com/yourusername/research-paper-summarizer/issues)
2. Review Streamlit documentation
3. Check Arxiv API status
4. Verify your Python environment

## 🔄 Development Workflow

### Making Changes
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
streamlit run main.py

# Commit changes
git add .
git commit -m "Add new feature"

# Push and create PR
git push origin feature/new-feature
```

### Code Quality
```bash
# Format code (optional)
pip install black
black *.py utils/*.py

# Check for issues
pip install flake8
flake8 *.py utils/*.py
```

---

**🎉 You're all set! Happy researching!**