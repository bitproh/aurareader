"""
summarizer.py - 100% Offline Extractive Summarization using TextRank/LexRank.
"""

from typing import List
import re
import math
from collections import Counter


def _fallback_tfidf_summarize(text: str, sentence_count: int = 5) -> List[str]:
    """Lightweight TF-IDF graph ranker as robust zero-dependency fallback."""
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 15]
    if not raw_sentences:
        return []
    if len(raw_sentences) <= sentence_count:
        return raw_sentences

    # Tokenize words per sentence
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'that', 'this', 'these', 'those', 'it', 'its', 'as', 'if', 'from', 'not', 'no', 'can', 'will'
    }
    
    sent_tokens = []
    doc_freq = Counter()
    for s in raw_sentences:
        tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', s) if w.lower() not in stop_words]
        sent_tokens.append(tokens)
        for t in set(tokens):
            doc_freq[t] += 1

    total_sents = len(raw_sentences)
    scores = []
    for idx, tokens in enumerate(sent_tokens):
        if not tokens:
            scores.append((0.0, idx))
            continue
        term_freq = Counter(tokens)
        score = 0.0
        for word, count in term_freq.items():
            idf = math.log((total_sents + 1) / (doc_freq[word] + 1)) + 1
            score += (count / len(tokens)) * idf
        # Slight boost to earlier sentences (e.g., intro of chapter)
        position_weight = 1.0 + (0.3 * (1.0 - (idx / total_sents)))
        scores.append((score * position_weight, idx))

    scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = sorted([idx for _, idx in scores[:sentence_count]])
    return [raw_sentences[i] for i in top_indices]


def summarize_text(text: str, sentence_count: int = 5) -> List[str]:
    """
    Summarizes text into key extractive sentences using TextRank / LexRank.
    Returns a list of top sentences.
    """
    if not text or len(text.strip()) < 50:
        return [text.strip()] if text else []

    try:
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        result = [str(s).strip() for s in summary_sentences if str(s).strip()]
        if result:
            return result
    except Exception:
        pass

    # Fallback to internal TF-IDF ranker
    return _fallback_tfidf_summarize(text, sentence_count=sentence_count)


def format_summary_bullet_points(sentences: List[str]) -> str:
    """Formats list of summary sentences into clean Markdown bullets."""
    if not sentences:
        return "No summary could be generated for this section."
    bullets = [f"• {s}" for s in sentences]
    return "\n\n".join(bullets)

