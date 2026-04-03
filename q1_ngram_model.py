"""
Q1. Form tokenization and filter stop words & punctuation.

This file demonstrates three tokenization approaches on Data_1.txt:
1. split()
2. Regular expression tokenization
3. NLTK word_tokenize

The file then uses the NLTK tokens as the group's preferred approach to:
- identify stop words found in the corpus
- identify punctuation found in the corpus
- remove stop words and punctuation
- print the filtered output

Report note:
The explanation in the report should state that NLTK word_tokenize is the
most suitable choice here because it separates punctuation cleanly while
preserving natural word boundaries better than split().
"""

import re
import string
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize


DATA_PATH = Path(__file__).resolve().parent / "Part A Dataset" / "Data_1.txt"
FALLBACK_STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "be",
    "each",
    "for",
    "from",
    "in",
    "is",
    "it",
    "may",
    "of",
    "other",
    "the",
    "to",
}


def load_stop_words():
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOP_WORDS


def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def remove_punctuation_and_stop_words(tokens, stop_words):
    cleaned_tokens = []
    removed_stop_words = []
    removed_punctuation = []

    for token in tokens:
        lower_token = token.lower()

        if token in string.punctuation:
            removed_punctuation.append(token)
            continue

        if lower_token in stop_words:
            removed_stop_words.append(token)
            continue

        cleaned_tokens.append(token)

    return cleaned_tokens, removed_stop_words, removed_punctuation


def q1_tokenization_and_filtering():
    text = DATA_PATH.read_text(encoding="utf-8").strip()
    ensure_nltk_resource("tokenizers/punkt", "punkt")

    # The three outputs below directly support the Q1 comparison in the report.
    split_tokens = text.split()
    regex_tokens = re.findall(r"\w+(?:-\w+)*(?:'\w+)?|[^\w\s]", text)
    nltk_tokens = word_tokenize(text)

    print("--- Q1.1 Tokenization Outputs ---")
    print("Original text:")
    print(text)
    print()

    print("1. split() tokenization")
    print(split_tokens)
    print(f"Token count: {len(split_tokens)}\n")

    print("2. Regular Expression tokenization")
    print(regex_tokens)
    print(f"Token count: {len(regex_tokens)}\n")

    print("3. NLTK tokenization")
    print(nltk_tokens)
    print(f"Token count: {len(nltk_tokens)}\n")

    print("--- Q1.2 Suitable Tokenization Justification ---")
    print(
        "NLTK tokenization is the most suitable choice for this corpus because it "
        "keeps punctuation as separate tokens while preserving the original words."
    )
    print(
        "split() leaves punctuation attached to words such as 'input.' and "
        "'variants.', while the regex version removes spacing well but is less "
        "standardized than NLTK for later NLP tasks."
    )
    print()

    stop_words = load_stop_words()
    # Stop-word and punctuation filtering is demonstrated using the preferred
    # tokenization output so the cleaned result is consistent and easier to justify.
    filtered_tokens, stop_words_found, punctuation_found = remove_punctuation_and_stop_words(
        nltk_tokens,
        stop_words,
    )

    unique_stop_words_found = list(dict.fromkeys(token.lower() for token in stop_words_found))
    unique_punctuation_found = list(dict.fromkeys(punctuation_found))

    print("--- Q1.3 Stop Words and Punctuation Removal ---")
    print("Stop words found in the corpus:")
    print(unique_stop_words_found)
    print()

    print("Punctuation found in the corpus:")
    print(unique_punctuation_found)
    print()

    print("Filtered tokens:")
    print(filtered_tokens)
    print()

    print("Filtered text:")
    print(" ".join(filtered_tokens))
    print()

    print("--- Q1.4 Importance of Filtering ---")
    print(
        "Removing stop words reduces very common words that add little meaning to "
        "classification or similarity tasks."
    )
    print(
        "Removing punctuation standardizes the corpus and prevents symbols from "
        "being treated as meaningful content features."
    )
    print(
        "Together, both steps help the model focus on informative content words "
        "such as 'classification', 'labels', 'variants', and 'sequence'."
    )


if __name__ == "__main__":
    q1_tokenization_and_filtering()
