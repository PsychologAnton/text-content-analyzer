#!/usr/bin/env python3
"""
Unit Tests for Text Content Analyzer
=====================================
Run with: python -m pytest test_analyzer.py -v
Or simply: python test_analyzer.py
"""

import unittest
import os
import tempfile
import shutil
from text_analyzer import TextAnalyzer


class TestTextAnalyzer(unittest.TestCase):
    """Test suite for TextAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test CSV file
        cls.csv_path = os.path.join(cls.test_dir, 'test_data.csv')
        with open(cls.csv_path, 'w', encoding='utf-8') as f:
            f.write('id;text;category\n')
            f.write('1;The quick brown fox jumps over the lazy dog;animal\n')
            f.write('2;Python programming language is powerful and flexible;tech\n')
            f.write('3;Machine learning algorithms process data efficiently;tech\n')
            f.write('4;The dog and fox became good friends in the forest;animal\n')
            f.write('5;Natural language processing uses machine learning;tech\n')
        
        # Create test TXT file
        cls.txt_path = os.path.join(cls.test_dir, 'test_data.txt')
        with open(cls.txt_path, 'w', encoding='utf-8') as f:
            f.write('The quick brown fox jumps over the lazy dog. ')
            f.write('Python programming language is powerful. ')
            f.write('Machine learning processes data efficiently. ')
            f.write('Natural language processing uses algorithms.')
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        shutil.rmtree(cls.test_dir)
    
    def test_1_csv_loading(self):
        """
        Test 1: Verify CSV file loading and token extraction.
        Should correctly load CSV, auto-detect text column, and extract tokens.
        """
        analyzer = TextAnalyzer()
        analyzer.load_file(self.csv_path, file_format='csv')
        
        # Check that tokens were extracted
        self.assertGreater(len(analyzer.tokens), 0, "Should extract tokens from CSV")
        self.assertGreater(len(analyzer.lemmas), 0, "Should create lemmas")
        
        # Verify stopwords were removed
        stopwords = ['the', 'and', 'is', 'over']
        for sw in stopwords:
            self.assertNotIn(sw, analyzer.tokens, f"Stopword '{sw}' should be removed")
        
        print(f"✓ Test 1 passed: CSV loading extracted {len(analyzer.tokens)} tokens")
    
    def test_2_txt_loading(self):
        """
        Test 2: Verify TXT file loading.
        Should correctly load plain text and process it.
        """
        analyzer = TextAnalyzer()
        analyzer.load_file(self.txt_path, file_format='txt')
        
        # Check that tokens were extracted
        self.assertGreater(len(analyzer.tokens), 0, "Should extract tokens from TXT")
        
        # Verify some expected words are present
        expected_words = ['quick', 'brown', 'fox', 'python', 'machine', 'learning']
        found = sum(1 for w in expected_words if w in analyzer.tokens)
        self.assertGreater(found, 0, "Should find expected content words")
        
        print(f"✓ Test 2 passed: TXT loading extracted {len(analyzer.tokens)} tokens")
    
    def test_3_metrics_calculation(self):
        """
        Test 3: Verify metrics calculation (N, V, TTR).
        Should correctly calculate tokens, vocabulary, and TTR.
        """
        analyzer = TextAnalyzer()
        analyzer.load_file(self.csv_path, file_format='csv')
        stats = analyzer.calculate_metrics()
        
        # Check all required metrics exist
        self.assertIn('N', stats, "Should have Tokens (N)")
        self.assertIn('V', stats, "Should have Vocabulary (V)")
        self.assertIn('TTR', stats, "Should have TTR")
        self.assertIn('Top20', stats, "Should have Top20 words")
        
        # Verify metric values are valid
        self.assertGreater(stats['N'], 0, "N should be positive")
        self.assertGreater(stats['V'], 0, "V should be positive")
        self.assertGreaterEqual(stats['V'], 1, "V should be at least 1")
        self.assertLessEqual(stats['V'], stats['N'], "V should be <= N")
        
        # Verify TTR calculation
        expected_ttr = round(stats['V'] / stats['N'], 4)
        self.assertEqual(stats['TTR'], expected_ttr, "TTR should equal V/N")
        self.assertGreater(stats['TTR'], 0, "TTR should be positive")
        self.assertLessEqual(stats['TTR'], 1, "TTR should be <= 1")
        
        print(f"✓ Test 3 passed: Metrics calculated - N={stats['N']}, V={stats['V']}, TTR={stats['TTR']:.4f}")
    
    def test_4_top20_words(self):
        """
        Test 4: Verify Top 20 word frequency list.
        Should return correctly sorted word frequencies.
        """
        analyzer = TextAnalyzer()
        analyzer.load_file(self.csv_path, file_format='csv')
        stats = analyzer.calculate_metrics()
        
        top20 = stats['Top20']
        
        # Check structure
        self.assertIsInstance(top20, list, "Top20 should be a list")
        self.assertLessEqual(len(top20), 20, "Top20 should have max 20 items")
        
        # Verify each item is (word, frequency) tuple
        for item in top20:
            self.assertIsInstance(item, tuple, "Each item should be a tuple")
            self.assertEqual(len(item), 2, "Each tuple should have 2 elements")
            word, freq = item
            self.assertIsInstance(word, str, "Word should be string")
            self.assertIsInstance(freq, int, "Frequency should be int")
            self.assertGreater(freq, 0, "Frequency should be positive")
        
        # Verify sorted by frequency (descending)
        frequencies = [freq for _, freq in top20]
        self.assertEqual(frequencies, sorted(frequencies, reverse=True), 
                        "Top20 should be sorted by frequency descending")
        
        print(f"✓ Test 4 passed: Top {len(top20)} words correctly sorted")
    
    def test_5_report_generation(self):
        """
        Test 5: Verify HTML report generation.
        Should create valid HTML file with all required content.
        """
        analyzer = TextAnalyzer()
        analyzer.load_file(self.csv_path, file_format='csv')
        analyzer.calculate_metrics()
        
        # Generate report
        report_path = os.path.join(self.test_dir, 'test_report.html')
        result_path = analyzer.generate_report(report_path)
        
        # Check file was created
        self.assertTrue(os.path.exists(result_path), "Report file should be created")
        self.assertEqual(result_path, report_path, "Should return correct path")
        
        # Read and verify content
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check required elements are present
        required_elements = [
            '<!DOCTYPE html>',
            'Tokens (N)',
            'Vocabulary (V)',
            'TTR',
            'Top 20 Words',
            str(analyzer.stats['N']),
            str(analyzer.stats['V']),
        ]
        
        for element in required_elements:
            self.assertIn(element, html_content, 
                         f"Report should contain '{element}'")
        
        # Clean up
        os.remove(report_path)
        
        print(f"✓ Test 5 passed: HTML report generated successfully")


def run_tests():
    """Run all tests with verbose output."""
    print("\n" + "="*60)
    print("TEXT CONTENT ANALYZER - TEST SUITE")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTextAnalyzer)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
