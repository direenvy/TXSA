"""
Q3. Form POS taggers and syntactic analysers.

This file uses Data_2.txt to demonstrate:
1. NLTK POS tagging
2. TextBlob POS tagging
3. Regexp POS tagging
4. Parse tree generation with a context-free grammar

Report note:
NLTK and TextBlob are trained taggers, while the Regexp tagger is rule-based.
The parse tree section shows one valid grammatical analysis of the sentence.
"""

from pathlib import Path

import nltk
from nltk import CFG, ChartParser
from nltk.tag import RegexpTagger
from nltk.tokenize import TreebankWordTokenizer
from textblob import TextBlob
from textblob.exceptions import MissingCorpusError


DATA_PATH = Path(__file__).resolve().parent / "Part A Dataset" / "Data_2.txt"


def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def q3_pos_parsing():
    text = DATA_PATH.read_text(encoding="utf-8").strip()
    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)

    # NLTK POS tagging depends on the perceptron tagger resource.
    ensure_nltk_resource("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng")
    ensure_nltk_resource("tokenizers/punkt", "punkt")

    print("--- Q3.1 POS Tagging ---")
    print(f"Sentence: {text}")
    print(f"Tokens: {tokens}\n")

    nltk_tags = nltk.pos_tag(tokens)

    try:
        textblob_tags = TextBlob(text).tags
    except MissingCorpusError:
        textblob_tags = [("TEXTBLOB_RESOURCE_MISSING", "N/A")]

    # The Regexp tagger is intentionally transparent so the report can explain
    # how rule-based tagging differs from trained taggers.
    patterns = [
        (r"^(The|the|a|an)$", "DT"),
        (r"^(big|black|white)$", "JJ"),
        (r"^(dog|cat)$", "NN"),
        (r"^(barked|chased)$", "VBD"),
        (r"^(at)$", "IN"),
        (r"^(and)$", "CC"),
        (r"^(away)$", "RB"),
        (r"^\.$", "."),
        (r".*", "NN"),
    ]
    regexp_tags = RegexpTagger(patterns).tag(tokens)

    print("1. NLTK POS tags")
    print(nltk_tags)
    print()

    print("2. TextBlob POS tags")
    print(textblob_tags)
    print()

    print("3. Regexp POS tags")
    print(regexp_tags)
    print()

    textblob_map = {token: tag for token, tag in textblob_tags}

    print("--- Q3.2 POS Tagger Comparison ---")
    print(f"{'Token':<12} {'NLTK':<10} {'TextBlob':<10} {'Regexp':<10}")
    for token, nltk_tag in nltk_tags:
        print(
            f"{token:<12} "
            f"{nltk_tag:<10} "
            f"{textblob_map.get(token, 'N/A'):<10} "
            f"{dict(regexp_tags).get(token, 'N/A'):<10}"
        )

    print("\nComparison notes:")
    print("NLTK and TextBlob are trained taggers, so they rely on learned language patterns.")
    print("The Regexp tagger is rule-based, so its output depends fully on the patterns we define.")
    print("Regexp tagging is transparent and easy to explain, but it is less flexible than trained taggers.")

    print("\n--- Q3.3 Parse Trees ---")
    # The grammar is tailored to the given sentence so the parser can build
    # a valid tree for the required syntactic analysis.
    grammar = CFG.fromstring(
        """
        S -> NP VP PUNCT
        VP -> VP CONJ VP | V PP | V ADV
        NP -> DET ADJ ADJ N | DET ADJ N | DET N
        PP -> P NP

        DET -> 'The' | 'the'
        ADJ -> 'big' | 'black' | 'white'
        N -> 'dog' | 'cat'
        V -> 'barked' | 'chased'
        P -> 'at'
        CONJ -> 'and'
        ADV -> 'away'
        PUNCT -> '.'
        """
    )
    parser = ChartParser(grammar)
    trees = list(parser.parse(tokens))

    if not trees:
        print("No parse tree was generated for the sentence.")
        return

    for index, tree in enumerate(trees, start=1):
        print(f"Parse tree {index}:")
        print(tree)
        tree.pretty_print()


if __name__ == "__main__":
    q3_pos_parsing()
