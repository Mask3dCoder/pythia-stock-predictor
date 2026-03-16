"""
Sentiment Analyzer Module

Provides sentiment analysis using VADER, TextBlob, and BERT models.
"""

import logging
import re
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment from text data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Sentiment Analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.vader_available = True
        except ImportError:
            logger.warning("VADER not available. Install vaderSentiment for sentiment analysis.")
            self.vader = None
            self.vader_available = False
            
        # Initialize TextBlob
        try:
            from textblob import TextBlob
            self.textblob = TextBlob
            self.textblob_available = True
        except ImportError:
            logger.warning("TextBlob not available. Install textblob for sentiment analysis.")
            self.textblob = None
            self.textblob_available = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text data.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_vader(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.vader_available:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 0.0,
                'sentiment': 'neutral'
            }
            
        scores = self.vader.polarity_scores(text)
        
        # Determine sentiment category
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu'],
            'sentiment': sentiment
        }
    
    def analyze_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.textblob_available:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'sentiment': 'neutral'
            }
            
        blob = self.textblob(text)
        
        # Determine sentiment category
        if blob.sentiment.polarity > 0.05:
            sentiment = 'positive'
        elif blob.sentiment.polarity < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment': sentiment
        }
    
    def analyze_text(self, text: str, method: str = 'vader') -> Dict:
        """
        Analyze sentiment using specified method.
        
        Args:
            text: Text to analyze
            method: 'vader', 'textblob', or 'combined'
            
        Returns:
            Dictionary with sentiment scores
        """
        cleaned_text = self.clean_text(text)
        
        if method == 'vader':
            return self.analyze_vader(cleaned_text)
        elif method == 'textblob':
            return self.analyze_textblob(cleaned_text)
        elif method == 'combined':
            vader_result = self.analyze_vader(cleaned_text)
            textblob_result = self.analyze_textblob(cleaned_text)
            
            # Average the scores
            combined_compound = (vader_result['compound'] + textblob_result['polarity']) / 2
            
            if combined_compound >= 0.05:
                sentiment = 'positive'
            elif combined_compound <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                'compound': combined_compound,
                'vader_compound': vader_result['compound'],
                'textblob_polarity': textblob_result['polarity'],
                'subjectivity': textblob_result['subjectivity'],
                'sentiment': sentiment
            }
        else:
            logger.warning(f"Unknown method: {method}, using vader")
            return self.analyze_vader(cleaned_text)
    
    def analyze_batch(self, texts: List[str], method: str = 'vader') -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            method: 'vader', 'textblob', or 'combined'
            
        Returns:
            DataFrame with sentiment results
        """
        results = []
        
        for i, text in enumerate(texts):
            result = self.analyze_text(text, method)
            result['text'] = text[:100]  # Store first 100 chars
            result['index'] = i
            results.append(result)
            
        df = pd.DataFrame(results)
        
        logger.info(f"Analyzed {len(texts)} texts using {method}")
        
        return df
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        method: str = 'vader'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame.
        
        Args:
            df: DataFrame with text data
            text_column: Column name containing text
            method: 'vader', 'textblob', or 'combined'
            
        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()
        
        # Apply sentiment analysis
        if method == 'vader':
            sentiment_results = df[text_column].apply(self.analyze_vader)
            df['vader_compound'] = sentiment_results.apply(lambda x: x['compound'])
            df['vader_pos'] = sentiment_results.apply(lambda x: x['pos'])
            df['vader_neg'] = sentiment_results.apply(lambda x: x['neg'])
            df['vader_neu'] = sentiment_results.apply(lambda x: x['neu'])
            df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
            
        elif method == 'textblob':
            sentiment_results = df[text_column].apply(self.analyze_textblob)
            df['textblob_polarity'] = sentiment_results.apply(lambda x: x['polarity'])
            df['textblob_subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
            df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
            
        elif method == 'combined':
            sentiment_results = df[text_column].apply(lambda x: self.analyze_text(x, 'combined'))
            df['sentiment_compound'] = sentiment_results.apply(lambda x: x['compound'])
            df['vader_compound'] = sentiment_results.apply(lambda x: x['vader_compound'])
            df['textblob_polarity'] = sentiment_results.apply(lambda x: x['textblob_polarity'])
            df['sentiment'] = sentiment_results.apply(lambda x: x['sentiment'])
            
        logger.info(f"Analyzed DataFrame sentiment: {len(df)} rows")
        
        return df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of sentiment data.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_items': len(df),
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0
        }
        
        if 'sentiment' in df.columns:
            summary['positive_count'] = len(df[df['sentiment'] == 'positive'])
            summary['negative_count'] = len(df[df['sentiment'] == 'negative'])
            summary['neutral_count'] = len(df[df['sentiment'] == 'neutral'])
            
            summary['positive_pct'] = summary['positive_count'] / summary['total_items'] * 100
            summary['negative_pct'] = summary['negative_count'] / summary['total_items'] * 100
            summary['neutral_pct'] = summary['neutral_count'] / summary['total_items'] * 100
            
        if 'compound' in df.columns:
            summary['mean_compound'] = df['compound'].mean()
            summary['std_compound'] = df['compound'].std()
            
        if 'polarity' in df.columns:
            summary['mean_polarity'] = df['polarity'].mean()
            summary['std_polarity'] = df['polarity'].std()
            
        return summary
    
    def create_sentiment_features(
        self,
        df: pd.DataFrame,
        text_column: str,
        date_column: Optional[str] = None,
        method: str = 'vader'
    ) -> pd.DataFrame:
        """
        Create time-based sentiment features for stock data.
        
        Args:
            df: DataFrame with stock data and text
            text_column: Column with news/headlines
            date_column: Column with dates (if None, use index)
            method: Sentiment analysis method
            
        Returns:
            DataFrame with aggregated sentiment features
        """
        df = df.copy()
        
        # Analyze sentiment
        df = self.analyze_dataframe(df, text_column, method)
        
        # Aggregate by date if date column exists
        if date_column:
            df[date_column] = pd.to_datetime(df[date_column])
            
            agg_dict = {}
            if 'compound' in df.columns:
                agg_dict['compound'] = ['mean', 'std', 'min', 'max']
            if 'polarity' in df.columns:
                agg_dict['polarity'] = ['mean', 'std', 'min', 'max']
                
            daily_sentiment = df.groupby(date_column).agg(agg_dict)
            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
            daily_sentiment = daily_sentiment.reset_index()
            
            return daily_sentiment
            
        return df
    
    def get_word_importance(
        self,
        texts: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get most important words based on sentiment association.
        
        Args:
            texts: List of text documents
            top_n: Number of top words to return
            
        Returns:
            DataFrame with word frequencies and sentiment
        """
        from collections import Counter
        
        # Analyze sentiment for each text
        sentiments = []
        all_words = []
        
        for text in texts:
            cleaned = self.clean_text(text)
            words = cleaned.split()
            
            sentiment = self.analyze_vader(cleaned)['compound']
            
            for word in words:
                if len(word) > 2:  # Skip very short words
                    all_words.append((word, sentiment))
        
        # Aggregate by word
        word_sentiments = {}
        word_counts = Counter([w[0] for w in all_words])
        
        for word, sent in all_words:
            if word not in word_sentiments:
                word_sentiments[word] = []
            word_sentiments[word].append(sent)
        
        # Calculate statistics
        word_stats = []
        for word, sentiments_list in word_sentiments.items():
            if word_counts[word] >= 3:  # Minimum frequency
                word_stats.append({
                    'word': word,
                    'count': word_counts[word],
                    'avg_sentiment': np.mean(sentiments_list),
                    'std_sentiment': np.std(sentiments_list)
                })
        
        df = pd.DataFrame(word_stats)
        df = df.sort_values('count', ascending=False).head(top_n)
        
        return df
