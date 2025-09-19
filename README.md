<<<<<<< Updated upstream
# Sentiment-Analyzer
=======

# Sentiment Analyzer

An advanced sentiment analysis web application built with Streamlit and Hugging Face's API. It allows users to analyze the sentiment of single texts, batch datasets, compare datasets, generate accuracy reports, and export results in multiple formats.

## Features

- **Single Text Analysis:** Enter any text and get its sentiment (positive, negative, neutral), confidence score, keywords, and probability distribution.
- **Batch Analysis:** Upload a CSV file and analyze sentiment for each row. Visualize sentiment distribution and download results.
- **Dataset Comparison:** Compare sentiment distributions between two datasets with multiple chart types (bar, pie, histogram, line) and download detailed results.
- **Accuracy Report:** Generate classification reports and confusion matrices using labeled data or comparison results.
- **Export Results:** Download analysis results in CSV, Excel, HTML, Word, PDF, and JSON formats.
- **Customizable Models:** Choose from multiple Hugging Face models for sentiment analysis.
- **Keyword Extraction:** Extract top keywords from each text using NLP techniques.
- **Beautiful UI:** Modern, responsive interface with custom CSS styling and interactive charts.

## Technologies Used

- Python 3
- Streamlit (UI framework)
- Hugging Face Inference API
- scikit-learn (metrics, vectorization)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- NLTK (tokenization, stopwords)
- xhtml2pdf, python-docx, xlsxwriter (export)

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/Lutho123-Pe/My-Work.git
	cd My-Work
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

## Usage

Run the Streamlit app:
```bash
streamlit run sentiment_analyzer.py
```

### Single Text Analysis
- Enter text in the UI and click "Analyze text" to view sentiment, confidence, keywords, and probability distribution.

### Batch Analysis
- Upload a CSV file with a text column. Select the column and run batch analysis. View and download results and visualizations.

### Compare Datasets
- Upload two CSV files, select text columns, and compare sentiment distributions with interactive charts. Download comparison results.

### Accuracy Report
- Upload labeled data or use comparison results to generate classification reports and confusion matrices.

### Export Results
- Download results in CSV, Excel, HTML, Word, PDF, or JSON formats.

## Configuration

- **Model Selection:** Choose from multiple Hugging Face models in the sidebar.
- **API Key:** The app uses a pre-configured Hugging Face API key for real-time analysis.
- **Demo Mode:** Optionally enable demo mode for mock analysis (no API required).

## Requirements

All dependencies are listed in `requirements.txt`:

```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
huggingface_hub
requests
python-docx
xhtml2pdf
chardet
nltk
xlsxwriter
```

## File Structure

- `sentiment_analyzer.py` — Main Streamlit app
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## License

This project is licensed under the MIT License.
# Sentiment Analyzer
>>>>>>> Stashed changes
