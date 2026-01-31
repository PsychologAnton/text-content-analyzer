#!/usr/bin/env python3
"""
Text Content Analyzer
=====================
A tool for primary text data processing and content analysis.
Supports CSV and TXT input formats with console output or HTML report generation.

Features:
- Tokenization and English lemmatization
- Stop words removal
- TTR (Type-Token Ratio) calculation
- Top 20 word frequency analysis
- Word Cloud visualization
- Console and HTML report output modes

Usage:
    python text_analyzer.py <input_file> [--mode console|report] [--format csv|txt]
"""

import argparse
import os
import sys
import base64
from datetime import datetime
from collections import Counter
from io import BytesIO

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Optional: WordCloud
try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# NLTK setup
def setup_nltk():
    """Download required NLTK data if not present."""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

setup_nltk()


class TextAnalyzer:
    """
    Text Content Analyzer for corpus linguistics and content analysis.
    
    Attributes:
        tokens (list): List of processed tokens
        lemmas (list): List of lemmatized tokens
        stats (dict): Calculated statistics (N, V, TTR, Top20)
    """
    
    def __init__(self):
        self.tokens = []
        self.lemmas = []
        self.stats = {}
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.wordcloud_base64 = None
    
    def load_file(self, filepath: str, file_format: str = 'csv', 
                  text_column: str = None, delimiter: str = ';') -> None:
        """
        Load and process text data from file.
        
        Args:
            filepath: Path to input file
            file_format: 'csv' or 'txt'
            text_column: Column name for CSV (auto-detected if None)
            delimiter: CSV delimiter character
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if file_format.lower() == 'csv':
            self._load_csv(filepath, text_column, delimiter)
        elif file_format.lower() == 'txt':
            self._load_txt(filepath)
        else:
            raise ValueError(f"Unsupported format: {file_format}. Use 'csv' or 'txt'.")
    
    def _load_csv(self, filepath: str, text_column: str, delimiter: str) -> None:
        """Load text from CSV file."""
        df = pd.read_csv(filepath, sep=delimiter, encoding='utf-8', on_bad_lines='skip')
        
        # Auto-detect text column
        if text_column is None:
            candidates = ['text', 'content', 'statement', 'utterance', 'message', 'body']
            text_column = next((col for col in candidates if col in df.columns), None)
            if text_column is None and len(df.columns) > 0:
                text_column = df.columns[0]
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
        
        raw_text = " ".join(df[text_column].dropna().astype(str))
        self._process_text(raw_text)
    
    def _load_txt(self, filepath: str) -> None:
        """Load text from TXT file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        self._process_text(raw_text)
    
    def _process_text(self, text: str) -> None:
        """
        Process raw text: tokenize, clean, and lemmatize.
        
        Args:
            text: Raw text string
        """
        # Tokenization
        raw_tokens = word_tokenize(text.lower())
        
        # Clean tokens: alphabetic, length > 2, not stopwords
        self.tokens = [
            w for w in raw_tokens 
            if w.isalpha() and len(w) > 2 and w not in self.stop_words
        ]
        
        # Lemmatization (English)
        self.lemmas = [self.lemmatizer.lemmatize(token) for token in self.tokens]
    
    def calculate_metrics(self) -> dict:
        """
        Calculate text statistics.
        
        Returns:
            dict with keys: N (tokens), V (vocabulary), TTR, Top20
        """
        if not self.lemmas:
            raise ValueError("No data loaded. Call load_file() first.")
        
        freq_dist = Counter(self.lemmas)
        
        n_tokens = len(self.lemmas)
        n_vocabulary = len(set(self.lemmas))
        ttr = round(n_vocabulary / n_tokens, 4) if n_tokens > 0 else 0
        
        self.stats = {
            'N': n_tokens,           # Total tokens
            'V': n_vocabulary,       # Vocabulary size (unique lemmas)
            'TTR': ttr,              # Type-Token Ratio
            'Top20': freq_dist.most_common(20)
        }
        
        return self.stats
    
    def generate_wordcloud(self) -> str:
        """
        Generate WordCloud image as base64 string.
        
        Returns:
            Base64 encoded PNG image string, or None if wordcloud unavailable
        """
        if not WORDCLOUD_AVAILABLE:
            return None
        
        if not self.lemmas:
            return None
        
        # Create word cloud
        text = " ".join(self.lemmas)
        wc = WordCloud(
            width=1000,
            height=500,
            background_color='white',
            max_words=100,
            colormap='Blues',
            prefer_horizontal=0.7,
            min_font_size=10
        ).generate(text)
        
        # Save to base64
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        self.wordcloud_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()
        
        return self.wordcloud_base64
    
    def print_console(self) -> None:
        """
        Print statistics to console.
        """
        if not self.stats:
            self.calculate_metrics()
        
        print("\n" + "="*60)
        print("TEXT CONTENT ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nBASIC METRICS:")
        print(f"   Tokens (N):      {self.stats['N']:,}")
        print(f"   Vocabulary (V):  {self.stats['V']:,}")
        print(f"   TTR:             {self.stats['TTR']:.4f}")
        
        print(f"\nTOP 20 WORDS (Lemmatized):")
        print("-"*40)
        print(f"{'Rank':<6}{'Word':<20}{'Frequency':<12}")
        print("-"*40)
        
        for rank, (word, freq) in enumerate(self.stats['Top20'], 1):
            print(f"{rank:<6}{word:<20}{freq:<12}")
        
        print("\n" + "="*60)
    
    def generate_report(self, output_path: str = 'report.html') -> str:
        """
        Generate HTML report with all statistics and WordCloud.
        
        Args:
            output_path: Path for output HTML file
            
        Returns:
            Path to generated report
        """
        if not self.stats:
            self.calculate_metrics()
        
        # Generate WordCloud
        self.generate_wordcloud()
        
        # Build top 20 table rows
        top20_rows = ""
        for rank, (word, freq) in enumerate(self.stats['Top20'], 1):
            rel_freq = freq / self.stats['N'] * 100
            top20_rows += f"""<tr>
                <td>{rank}</td>
                <td><strong>{word}</strong></td>
                <td>{freq:,}</td>
                <td>{rel_freq:.2f}%</td>
            </tr>\n"""
        
        # WordCloud section
        wordcloud_section = ""
        if self.wordcloud_base64:
            wordcloud_section = f"""
            <h2>Word Cloud</h2>
            <p class="description">Word size is proportional to frequency in the corpus.</p>
            <div class="wordcloud-container">
                <img src="data:image/png;base64,{self.wordcloud_base64}" alt="Word Cloud" class="wordcloud-img">
            </div>
            """
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Content Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif; 
            background: #f5f5f5;
            min-height: 100vh; 
            padding: 40px 20px;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .card {{
            background: white; 
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 40px; 
            margin-bottom: 24px;
        }}
        h1 {{ 
            color: #1a1a1a; 
            font-size: 1.8em; 
            margin-bottom: 8px; 
            font-weight: 600;
        }}
        h2 {{ 
            color: #333; 
            font-size: 1.2em; 
            margin: 32px 0 16px; 
            padding-bottom: 8px;
            border-bottom: 2px solid #e0e0e0; 
            font-weight: 600;
        }}
        .subtitle {{
            color: #666;
            font-size: 0.95em;
            margin-bottom: 24px;
        }}
        .description {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 16px;
        }}
        .metrics-grid {{ 
            display: grid; 
            grid-template-columns: repeat(3, 1fr); 
            gap: 16px; 
            margin: 20px 0; 
        }}
        .metric-box {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 24px; 
            border-radius: 8px; 
            text-align: center;
        }}
        .metric-box .value {{ 
            font-size: 2.2em; 
            font-weight: 700; 
            color: #1a1a1a;
        }}
        .metric-box .label {{ 
            font-size: 0.85em; 
            color: #666; 
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 16px; 
        }}
        th, td {{ 
            padding: 12px 16px; 
            text-align: left; 
            border-bottom: 1px solid #e9ecef; 
        }}
        th {{ 
            background: #f8f9fa; 
            color: #333; 
            font-weight: 600;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .wordcloud-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .wordcloud-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        .methodology {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-top: 16px;
        }}
        .methodology ul {{
            color: #555;
            padding-left: 20px;
        }}
        .methodology li {{
            margin: 8px 0;
        }}
        .footer {{ 
            text-align: center; 
            color: #999; 
            margin-top: 32px; 
            font-size: 0.85em; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Text Content Analysis Report</h1>
            <p class="subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Basic Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="value">{self.stats['N']:,}</div>
                    <div class="label">Tokens (N)</div>
                </div>
                <div class="metric-box">
                    <div class="value">{self.stats['V']:,}</div>
                    <div class="label">Vocabulary (V)</div>
                </div>
                <div class="metric-box">
                    <div class="value">{self.stats['TTR']:.4f}</div>
                    <div class="label">TTR</div>
                </div>
            </div>
            
            {wordcloud_section}
            
            <h2>Top 20 Words</h2>
            <p class="description">Most frequent lemmatized words in the corpus.</p>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Word</th>
                        <th>Frequency</th>
                        <th>Relative</th>
                    </tr>
                </thead>
                <tbody>
                    {top20_rows}
                </tbody>
            </table>
            
            <h2>Methodology</h2>
            <div class="methodology">
                <ul>
                    <li><strong>Tokenization:</strong> NLTK word_tokenize</li>
                    <li><strong>Lemmatization:</strong> WordNet Lemmatizer (English)</li>
                    <li><strong>Filtering:</strong> Alphabetic tokens, length > 2, stopwords removed</li>
                    <li><strong>TTR Formula:</strong> V / N (Vocabulary / Tokens)</li>
                </ul>
            </div>
        </div>
        <div class="footer">Text Content Analyzer</div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path


def main():
    """Command-line interface for Text Analyzer."""
    parser = argparse.ArgumentParser(
        description='Text Content Analyzer - Primary data processing for content analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python text_analyzer.py data.csv --mode console
  python text_analyzer.py corpus.txt --format txt --mode report
  python text_analyzer.py data.csv --column text --delimiter ','
        """
    )
    
    parser.add_argument('input_file', help='Path to input file (CSV or TXT)')
    parser.add_argument('--format', '-f', choices=['csv', 'txt'], default='csv',
                        help='Input file format (default: csv)')
    parser.add_argument('--mode', '-m', choices=['console', 'report'], default='console',
                        help='Output mode: console (print stats) or report (generate HTML)')
    parser.add_argument('--column', '-c', default=None,
                        help='Text column name for CSV (auto-detected if not specified)')
    parser.add_argument('--delimiter', '-d', default=';',
                        help='CSV delimiter (default: ;)')
    parser.add_argument('--output', '-o', default='report.html',
                        help='Output path for HTML report (default: report.html)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    try:
        # Load and process data
        print(f"\n>>> Loading {args.input_file}...")
        analyzer.load_file(
            filepath=args.input_file,
            file_format=args.format,
            text_column=args.column,
            delimiter=args.delimiter
        )
        print(f"    Loaded {len(analyzer.tokens):,} tokens")
        
        # Calculate metrics
        print(">>> Calculating metrics...")
        analyzer.calculate_metrics()
        print(f"    TTR = {analyzer.stats['TTR']:.4f}")
        
        # Output results
        if args.mode == 'console':
            analyzer.print_console()
        else:
            print(">>> Generating report...")
            report_path = analyzer.generate_report(args.output)
            print(f"\nReport generated: {report_path}")
            print(f"Open in browser to view results.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
