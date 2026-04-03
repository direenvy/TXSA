"""
Q2. Form word stemming.

This file demonstrates stemming on Data_1.txt using:
1. Regular Expression stemmer
2. Porter Stemmer
3. Lancaster Stemmer

It prints both sentence-level examples and word-level comparisons so the
report can explain how aggressive or conservative each stemmer is.

Report note:
Porter is usually the safest default because it balances normalization with
readability, while Lancaster is more aggressive and Regexp is the simplest.
"""

import re
from pathlib import Path

from nltk.stem import LancasterStemmer, PorterStemmer, RegexpStemmer


DATA_PATH = Path(__file__).resolve().parent / "Part A Dataset" / "Data_1.txt"


def q2_stemming_check():
    text = DATA_PATH.read_text(encoding="utf-8")
    tokens = re.findall(r"[A-Za-z']+", text.lower())

    # These three stemmers are required by the question paper.
    regexp_stemmer = RegexpStemmer(r"ing$|ed$|ly$|ies$|s$", min=4)
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()

    print("--- Q2.1 Importance of Stemming ---")
    print(
        "Stemming groups related word forms into a shared root so text analytics "
        "models can treat words such as 'classified' and 'classification' as related terms."
    )
    print(
        "This reduces vocabulary size and helps models focus on concept frequency "
        "instead of small grammatical variations."
    )
    print()

    print("--- Q2.2 Stemming Demonstration ---")
    print(f"Total tokens: {len(tokens)}")
    print(f"First 30 original tokens: {tokens[:30]}\n")

    print("Regular Expression stemmer (first 30)")
    print([regexp_stemmer.stem(token) for token in tokens[:30]])
    print()

    print("Porter stemmer (first 30)")
    print([porter_stemmer.stem(token) for token in tokens[:30]])
    print()

    print("Lancaster stemmer (first 30)")
    print([lancaster_stemmer.stem(token) for token in tokens[:30]])
    print()

    sample_words = [
        "classification",
        "choosing",
        "correct",
        "labels",
        "defined",
        "interesting",
        "variants",
        "multiclass",
        "multiple",
        "assigned",
        "sequence",
        "jointly",
        "classified",
    ]

    print("--- Q2.3 Stemmer Comparison ---")
    # The selected words make it easier to discuss the differences clearly in the report.
    print(f"{'Word':<18} {'Regexp':<18} {'Porter':<18} {'Lancaster':<18}")
    for word in sample_words:
        print(
            f"{word:<18} "
            f"{regexp_stemmer.stem(word):<18} "
            f"{porter_stemmer.stem(word):<18} "
            f"{lancaster_stemmer.stem(word):<18}"
        )

    print("\nComparison notes:")
    print(
        "The Regexp stemmer is the simplest because it only removes endings that "
        "match the specified pattern."
    )
    print(
        "Porter is more balanced and usually keeps a recognizable stem, so it is "
        "often the safest default for English text."
    )
    print(
        "Lancaster is the most aggressive stemmer here, which can shorten words "
        "more heavily and sometimes over-stem them."
    )


if __name__ == "__main__":
    q2_stemming_check()
