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
- Console and HTML report output modes

Usage:
    python text_analyzer.py <input_file> [--mode console|report] [--format csv|txt]
"""

import argparse
import os
import sys
from datetime import datetime
from collections import Counter

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
    
    def print_console(self) -> None:
        """
        Print statistics to console.
        """
        if not self.stats:
            self.calculate_metrics()
        
        print("\n" + "="*60)
        print("TEXT CONTENT ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📊 BASIC METRICS:")
        print(f"   Tokens (N):      {self.stats['N']:,}")
        print(f"   Vocabulary (V):  {self.stats['V']:,}")
        print(f"   TTR:             {self.stats['TTR']:.4f}")
        
        print(f"\n📝 TOP 20 WORDS (Lemmatized):")
        print("-"*40)
        print(f"{'Rank':<6}{'Word':<20}{'Frequency':<12}")
        print("-"*40)
        
        for rank, (word, freq) in enumerate(self.stats['Top20'], 1):
            print(f"{rank:<6}{word:<20}{freq:<12}")
        
        print("\n" + "="*60)
    
    def generate_report(self, output_path: str = 'report.html') -> str:
        """
        Generate HTML report with all statistics.
        
        Args:
            output_path: Path for output HTML file
            
        Returns:
            Path to generated report
        """
        if not self.stats:
            self.calculate_metrics()
        
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
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Content Analysis Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 40px 20px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        .card {{
            background: white; border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px; margin-bottom: 30px;
        }}
        h1 {{ color: #333; font-size: 2.2em; margin-bottom: 10px; }}
        h2 {{ color: #667eea; font-size: 1.4em; margin: 25px 0 15px; border-bottom: 2px solid #667eea; padding-bottom: 8px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 25px; border-radius: 12px; text-align: center;
        }}
        .metric-box .value {{ font-size: 2.5em; font-weight: bold; }}
        .metric-box .label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; color: #333; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .footer {{ text-align: center; color: rgba(255,255,255,0.8); margin-top: 30px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>📊 Text Content Analysis Report</h1>
            <p style="color: #666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
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
            
            <h2>Top 20 Words (Lemmatized)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Word</th>
                        <th>Frequency</th>
                        <th>Relative Freq.</th>
                    </tr>
                </thead>
                <tbody>
                    {top20_rows}
                </tbody>
            </table>
            
            <h2>Methodology</h2>
            <ul style="color: #666; line-height: 1.8; padding-left: 20px;">
                <li><strong>Tokenization:</strong> NLTK word_tokenize</li>
                <li><strong>Lemmatization:</strong> WordNet Lemmatizer (English)</li>
                <li><strong>Filtering:</strong> Alphabetic tokens, length > 2, stopwords removed</li>
                <li><strong>TTR Formula:</strong> V / N (Vocabulary / Tokens)</li>
            </ul>
        </div>
        <div class="footer">Text Content Analyzer | Content Analysis Tool</div>
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
        print(f"    ✓ Loaded {len(analyzer.tokens):,} tokens")
        
        # Calculate metrics
        print(">>> Calculating metrics...")
        analyzer.calculate_metrics()
        print(f"    ✓ TTR = {analyzer.stats['TTR']:.4f}")
        
        # Output results
        if args.mode == 'console':
            analyzer.print_console()
        else:
            report_path = analyzer.generate_report(args.output)
            print(f"\n✅ Report generated: {report_path}")
            print(f"   Open in browser to view results.")
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
