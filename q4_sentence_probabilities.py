import collections

def q4_sentence_probabilities():
    file_path = r"c:\Users\jrpha\OneDrive\Documents\txsa assignment\Part A Dataset\Data_3.txt"
    
    # 1. Load and Preprocess Data
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    
    training_data = []
    test_sentence = ""
    
    # Simple parsing logic based on file structure
    # Everything under "Training Corpus" until the next section
    # Everything under "Calculate sentence probability..." is test
    
    section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("~~"):
            continue
            
        if "Training Corpus" in line:
            section = "train"
            continue
        elif "Calculate sentence probability" in line:
            section = "test"
            continue
            
        if section == "train":
            training_data.append(line)
        elif section == "test":
            test_sentence = line

    print("--- 1. Data Loading ---")
    print(f"Training Sentences ({len(training_data)}):")
    for s in training_data:
        print(f"  {s}")
    print(f"\nTest Sentence:\n  {test_sentence}\n")

    # 2. Tokenization
    # We split by whitespace. Punctuation handling isn't specified, but <s> tags are space-separated.
    def tokenize(text):
        return text.split()

    train_tokens = [tokenize(s) for s in training_data]
    test_tokens = tokenize(test_sentence)
    
    # Vocabulary (V)
    # Collect all unique tokens from TRAINING data
    all_train_tokens = [t for sent in train_tokens for t in sent]
    vocab = sorted(list(set(all_train_tokens)))
    V = len(vocab)
    
    print(f"Vocabulary (V={V}): {vocab}\n")

    # 3. Model Training (Count Collection)
    unigram_counts = collections.Counter(all_train_tokens)
    bigram_counts = collections.defaultdict(int)
    
    for sent in train_tokens:
        for i in range(len(sent) - 1):
            w1 = sent[i]
            w2 = sent[i+1]
            bigram_counts[(w1, w2)] += 1
            
    # Prepare Bigrams for Test Sentence
    test_bigrams = [(test_tokens[i], test_tokens[i+1]) for i in range(len(test_tokens) - 1)]

    # --- Part 1 & 3: Unsmoothed Bigram Model ---
    print("--- Q4.1: Unsmoothed Bigram Model ---")
    print(f"{'Bigram (w1, w2)':<25} {'C(w1, w2)':<12} {'C(w1)':<10} {'Prob'}")
    
    unsmoothed_prob_total = 1.0
    
    for w1, w2 in test_bigrams:
        c_bigram = bigram_counts[(w1, w2)]
        c_unigram = unigram_counts[w1]
        
        if c_unigram == 0:
            prob = 0.0
        else:
            prob = c_bigram / c_unigram
            
        unsmoothed_prob_total *= prob
        print(f"{f'({w1}, {w2})':<25} {c_bigram:<12} {c_unigram:<10} {prob:.4f}")
        
    print(f"\nTotal Probability (Unsmoothed): {unsmoothed_prob_total}")
    # Format typically used: scientific notation if very small
    print(f"Result: {unsmoothed_prob_total:.6e}\n")


    # --- Part 2 & 3: Smoothed Bigram Model (Laplace / Add-One) ---
    print("--- Q4.2: Smoothed Bigram Model (Add-One) ---")
    print(f"Vocabulary Size V = {V}")
    print(f"{'Bigram (w1, w2)':<25} {'C(w1, w2)+1':<15} {'C(w1)+V':<12} {'Prob'}")
    
    smoothed_prob_total = 1.0
    
    for w1, w2 in test_bigrams:
        c_bigram = bigram_counts[(w1, w2)]
        c_unigram = unigram_counts[w1]
        
        # Add-One Smoothing Formula: P(w2|w1) = (C(w1, w2) + 1) / (C(w1) + V)
        numerator = c_bigram + 1
        denominator = c_unigram + V
        
        prob = numerator / denominator
        smoothed_prob_total *= prob
        
        print(f"{f'({w1}, {w2})':<25} {numerator:<15} {denominator:<12} {prob:.4f}")

    print(f"\nTotal Probability (Smoothed): {smoothed_prob_total}")
    print(f"Result: {smoothed_prob_total:.6e}")

if __name__ == "__main__":
    q4_sentence_probabilities()
