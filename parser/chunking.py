# requirements:
# pip install sentence-transformers faiss-cpu transformers nltk sentencepiece
import re
from typing import List
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def smart_chunk_text(text: str, max_tokens=500, overlap_tokens=100):
    # 1) Normalize headings (common patterns)
    # Split on headings like "1. Introduction", "4.3. Synthetic ECG image generation pipeline", or "Figure X."
    heading_pattern = re.compile(r'(^|\n)(\s{0,3}\d+(\.\d+)*\s+[\w ].+?:|\n[A-Z][A-Za-z0-9 \-]{4,100}\n)', re.M)
    # fallback: split by double newline
    sections = [s.strip() for s in re.split(r'\n{2,}', text) if s.strip()]
    # turn each section into token-aware chunks (using sentence tokenizer)
    chunks = []
    for sec in sections:
        sents = sent_tokenize(sec)
        cur = ""
        cur_tokens = 0
        for sent in sents:
            # approximate token count by whitespace; better: use tokenizer for exact
            toks = len(sent.split())
            if cur_tokens + toks > max_tokens:
                chunks.append(cur.strip())
                # start new with overlap
                if overlap_tokens > 0:
                    # keep last overlap_tokens words
                    words = cur.split()
                    overlap = " ".join(words[-overlap_tokens:]) if len(words) >= overlap_tokens else cur
                    cur = overlap + " " + sent
                    cur_tokens = len(cur.split())
                else:
                    cur = sent
                    cur_tokens = toks
            else:
                cur = (cur + " " + sent).strip()
                cur_tokens += toks
        if cur.strip():
            chunks.append(cur.strip())
    return chunks
