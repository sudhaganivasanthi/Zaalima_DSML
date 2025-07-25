# utils.py

import nltk
import spacy
import string
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in string.punctuation and token.text not in stop_words and not token.is_space
    ]
    return " ".join(tokens)
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        from pdfminer.high_level import extract_text
        return extract_text(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")




