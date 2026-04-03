"""
Q4. Work on sentence probabilities.

This file uses Data_3.txt to:
1. read the training corpus and test sentence
2. build unsmoothed bigram counts
3. build add-one smoothed bigram probabilities
4. print step-by-step values for each bigram in the test sentence

Report note:
The printed C(w1,w2), C(w1), C(w1,w2)+1, and C(w1)+V values can be used to
show the manual calculation steps in the report. The report still needs to
write those steps out explicitly because the question asks for manual work.
"""

from collections import Counter, defaultdict
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent / "Part A Dataset" / "Data_3.txt"


def parse_data_file():
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()

    training_sentences = []
    test_sentence = ""
    section = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("~"):
            continue

        if line == "Training Corpus":
            section = "train"
            continue

        if line.startswith("Calculate sentence probability"):
            section = "test"
            continue

        if section == "train":
            training_sentences.append(line)
        elif section == "test":
            test_sentence = line

    return training_sentences, test_sentence


def tokenize(sentence):
    return sentence.split()


def q4_sentence_probabilities():
    training_sentences, test_sentence = parse_data_file()

    train_tokens = [tokenize(sentence) for sentence in training_sentences]
    test_tokens = tokenize(test_sentence)

    all_train_tokens = [token for sentence in train_tokens for token in sentence]
    vocabulary = sorted(set(all_train_tokens))
    vocab_size = len(vocabulary)

    bigram_counts = defaultdict(int)
    context_counts = Counter()

    # Count each observed bigram and how often each first word appears as a context.
    for sentence in train_tokens:
        for index in range(len(sentence) - 1):
            w1 = sentence[index]
            w2 = sentence[index + 1]
            bigram_counts[(w1, w2)] += 1
            context_counts[w1] += 1

    test_bigrams = [(test_tokens[index], test_tokens[index + 1]) for index in range(len(test_tokens) - 1)]

    print("--- Q4 Data Overview ---")
    print("Training corpus:")
    for sentence in training_sentences:
        print(sentence)
    print()

    print(f"Test sentence: {test_sentence}")
    print(f"Vocabulary ({vocab_size} types): {vocabulary}\n")

    print("--- Q4.1 Unsmoothed Bigram Model ---")
    print(f"{'Bigram':<28} {'C(w1,w2)':<12} {'C(w1)':<10} {'P(w2|w1)':<12}")

    unsmoothed_probability = 1.0
    for w1, w2 in test_bigrams:
        count_bigram = bigram_counts[(w1, w2)]
        count_context = context_counts[w1]
        current_probability = count_bigram / count_context if count_context else 0.0
        unsmoothed_probability *= current_probability

        print(
            f"{f'({w1}, {w2})':<28} "
            f"{count_bigram:<12} "
            f"{count_context:<10} "
            f"{current_probability:<12.6f}"
        )

    print(f"Sentence probability (unsmoothed): {unsmoothed_probability:.10f}")
    print(f"Scientific notation: {unsmoothed_probability:.6e}\n")

    print("--- Q4.2 Smoothed Bigram Model (Add-One) ---")
    print(f"Vocabulary size V = {vocab_size}")
    print(f"{'Bigram':<28} {'C(w1,w2)+1':<14} {'C(w1)+V':<12} {'P_smooth':<12}")

    smoothed_probability = 1.0
    for w1, w2 in test_bigrams:
        count_bigram = bigram_counts[(w1, w2)]
        count_context = context_counts[w1]
        numerator = count_bigram + 1
        denominator = count_context + vocab_size
        current_probability = numerator / denominator
        smoothed_probability *= current_probability

        print(
            f"{f'({w1}, {w2})':<28} "
            f"{numerator:<14} "
            f"{denominator:<12} "
            f"{current_probability:<12.6f}"
        )

    print(f"Sentence probability (smoothed): {smoothed_probability:.10f}")
    print(f"Scientific notation: {smoothed_probability:.6e}")


if __name__ == "__main__":
    q4_sentence_probabilities()
