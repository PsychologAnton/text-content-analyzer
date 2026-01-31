# Text Content Analyzer

A Python tool for primary text data processing and content analysis. Designed for corpus linguistics, content analysis research, and text preprocessing tasks.

## Features

- **Tokenization** - Split text into meaningful units
- **English Lemmatization** - Reduce words to their base form using WordNet
- **Stop Words Removal** - Filter out common words
- **Frequency Analysis** - Calculate word frequencies and Top 20 words
- **Lexical Metrics** - Compute Tokens (N), Vocabulary (V), and TTR
- **Multiple Output Modes** - Console output or HTML report generation
- **Multiple Input Formats** - Support for CSV and TXT files

## Installation

```bash
# Clone the repository
git clone https://github.com/PsychologAnton/text-content-analyzer.git
cd text-content-analyzer

# Install dependencies
pip install -r requirements.txt
```

NLTK data will be downloaded automatically on first run.

## Usage

### Basic Usage

```bash
# Console output (default)
python text_analyzer.py sample_data/sample_corpus.csv

# Generate HTML report
python text_analyzer.py sample_data/sample_corpus.csv --mode report

# Process TXT file
python text_analyzer.py sample_data/sample_corpus.txt --format txt --mode console
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--format` | `-f` | Input file format: `csv` or `txt` | `csv` |
| `--mode` | `-m` | Output mode: `console` or `report` | `console` |
| `--column` | `-c` | Text column name for CSV | auto-detect |
| `--delimiter` | `-d` | CSV delimiter character | `;` |
| `--output` | `-o` | Output path for HTML report | `report.html` |

### Examples

```bash
# CSV with comma delimiter and specific column
python text_analyzer.py data.csv --delimiter ',' --column content --mode report

# Generate report with custom output path
python text_analyzer.py corpus.csv --mode report --output analysis_report.html

# Quick console analysis of text file
python text_analyzer.py document.txt -f txt -m console
```

## Output Metrics

| Metric | Description |
|--------|-------------|
| **Tokens (N)** | Total number of processed tokens |
| **Vocabulary (V)** | Number of unique lemmas |
| **TTR** | Type-Token Ratio (V/N) - measures lexical diversity |
| **Top 20 Words** | Most frequent lemmatized words with counts |

### Console Output Example

```
============================================================
TEXT CONTENT ANALYSIS RESULTS
============================================================

📊 BASIC METRICS:
   Tokens (N):      1,234
   Vocabulary (V):  456
   TTR:             0.3695

📝 TOP 20 WORDS (Lemmatized):
----------------------------------------
Rank  Word                Frequency   
----------------------------------------
1     analysis            45          
2     text                42          
3     language            38          
...
```

### HTML Report

The HTML report includes:
- Visual metrics cards with N, V, TTR
- Sortable Top 20 words table with relative frequencies
- Methodology description
- Professional styling

## Running Tests

```bash
# Run test suite
python test_analyzer.py

# Or using pytest
python -m pytest test_analyzer.py -v
```

### Test Coverage

| Test | Description |
|------|-------------|
| Test 1 | CSV file loading and token extraction |
| Test 2 | TXT file loading and processing |
| Test 3 | Metrics calculation (N, V, TTR) |
| Test 4 | Top 20 word frequency sorting |
| Test 5 | HTML report generation |

## Sample Data

The repository includes sample data for testing:

- `sample_data/sample_corpus.csv` - CSV format with text column
- `sample_data/sample_corpus.txt` - Plain text format

## Supported Input Formats

### CSV Format
- Semicolon (`;`) or comma (`,`) delimited
- Auto-detects text columns: `text`, `content`, `statement`, `utterance`, `message`, `body`
- UTF-8 encoding

### TXT Format
- Plain text files
- UTF-8 encoding
- Paragraphs separated by newlines

## Methodology

1. **Tokenization**: NLTK `word_tokenize`
2. **Filtering**: Keep alphabetic tokens with length > 2
3. **Stop Words**: English stop words removed (NLTK corpus)
4. **Lemmatization**: WordNet Lemmatizer
5. **TTR Calculation**: `TTR = V / N`

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- nltk >= 3.8.0

## License

MIT License

## Author

Created for content analysis research and text preprocessing tasks.
