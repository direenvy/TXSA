"""
Q5. Alternative approach implementation.

This file compares the group's Q1 tokenization choice, NLTK word_tokenize,
against an alternative tokenizer, NLTK ToktokTokenizer.

It prints:
1. the output of the group approach
2. the output of the alternative approach
3. a short compare-and-contrast explanation
4. a short judgment on whether the alternative is better, worse, or different

Report note:
For this corpus, ToktokTokenizer is a valid alternative, but word_tokenize
remains the stronger group choice because it gives cleaner punctuation handling.
"""

import re
from pathlib import Path

import nltk
from nltk.tokenize import ToktokTokenizer, word_tokenize


DATA_PATH = Path(__file__).resolve().parent / "Part A Dataset" / "Data_1.txt"


def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def q5_alternative_tokenization():
    text = DATA_PATH.read_text(encoding="utf-8").strip()

    ensure_nltk_resource("tokenizers/punkt", "punkt")

    # The group approach is the tokenizer selected in Q1.
    group_tokens = word_tokenize(text)
    alternative_tokenizer = ToktokTokenizer()
    alternative_tokens = alternative_tokenizer.tokenize(text)

    print("--- Q5.1 Alternative Tokenization Implementation ---")
    print("Group approach: NLTK word_tokenize")
    print("Alternative approach: NLTK ToktokTokenizer")
    print()

    print("Group tokenizer output:")
    print(group_tokens)
    print(f"Token count: {len(group_tokens)}\n")

    print("Alternative tokenizer output:")
    print(alternative_tokens)
    print(f"Token count: {len(alternative_tokens)}\n")

    print("--- Q5.2 Compare and Contrast ---")
    print(
        "Both approaches separate words from punctuation and are more suitable for "
        "text analytics than simple split-based tokenization."
    )
    print(
        "The group approach, word_tokenize, is a common general-purpose NLTK tokenizer "
        "and gives standard word-level tokenization for English text."
    )
    print(
        "The alternative ToktokTokenizer is rule-based and lightweight, and it does "
        "not depend on the same sentence-tokenization behavior as word_tokenize."
    )
    print()

    print("Tokens that contain punctuation or hyphen structure:")
    focus_tokens = [token for token in alternative_tokens if re.search(r"[.,;-]|-", token)]
    print(focus_tokens)
    print()

    print("--- Q5.3 Why It Is Better, Worse, or Different ---")
    print(
        "The alternative approach is different because ToktokTokenizer applies a "
        "simpler rule-based tokenization strategy."
    )
    print(
        "It can be better when you want a lightweight tokenizer with predictable "
        "punctuation handling and fewer resource dependencies."
    )
    print(
        "It can be worse than word_tokenize when you want the most standard NLTK "
        "English tokenization behavior for downstream NLP tasks."
    )
    print(
        "For this corpus, the group approach remains the stronger default, while "
        "ToktokTokenizer is still a valid alternative for comparison."
    )


if __name__ == "__main__":
    q5_alternative_tokenization()
