import collections

def q1_ngram_model():
    file_path = r"c:\Users\jrpha\OneDrive\Documents\txsa assignment\Part A Dataset\Data_1.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse the file content
    training_data = []
    test_sentence = ""
    
    mode = "none"
    for line in lines:
        line = line.strip()
        if not line or line.startswith("~~"):
            continue
            
        if line == "Training Corpus":
            mode = "train"
        elif line.startswith("Calculate sentence probability"):
            mode = "test"
        else:
            if mode == "train":
                training_data.append(line)
            elif mode == "test":
                test_sentence = line

    print(f"Training Data ({len(training_data)} sentences):")
    for s in training_data:
        print(f"  {s}")
    print(f"\nTest Sentence: {test_sentence}\n")

    # --- 1. Preprocessing ---
    # Tokenize (simple split by space)
    # Note: <s> and </s> are treated as tokens
    train_tokens = [s.split() for s in training_data]
    test_tokens = test_sentence.split()
    
    # Flatten training tokens for unigram counts
    all_train_tokens = [token for sent in train_tokens for token in sent]
    
    # --- 2. Unigram Model ---
    vocab = set(all_train_tokens)
    unigram_counts = collections.Counter(all_train_tokens)
    total_tokens = len(all_train_tokens)
    
    print("--- Unigram Counts & Probabilities ---")
    print(f"{'Token':<15} {'Count':<10} {'Probability':<10}")
    for token, count in unigram_counts.items():
        prob = count / total_tokens
        print(f"{token:<15} {count:<10} {prob:.4f}")
    
    # --- 3. Bigram Model ---
    bigram_counts = collections.defaultdict(int)
    bigram_context_counts = collections.defaultdict(int) # Count of w1 in (w1, w2)
    
    for sent in train_tokens:
        for i in range(len(sent) - 1):
            w1 = sent[i]
            w2 = sent[i+1]
            bigram_counts[(w1, w2)] += 1
            bigram_context_counts[w1] += 1
            
    print("\n--- Bigram Counts & Probabilities (MLE) ---")
    print(f"{'Bigram':<25} {'Count':<10} {'P(w2|w1)':<10}")
    
    # We will compute properties for the Bigrams found in the Test Sentence
    # To show the work for the specific task
    
    test_bigrams = []
    for i in range(len(test_tokens) - 1):
        test_bigrams.append((test_tokens[i], test_tokens[i+1]))
        
    sentence_prob = 1.0
    print("\n--- Calculation for Test Sentence ---")
    print(f"Test Sentence: {test_sentence}")
    print(f"{'Bigram':<25} {'Count(w1,w2)':<15} {'Count(w1)':<10} {'P(w2|w1)':<10}")
    
    for w1, w2 in test_bigrams:
        count_w1_w2 = bigram_counts[(w1, w2)]
        count_w1 = bigram_context_counts[w1]
        
        # MLE Probability
        if count_w1 > 0:
            prob = count_w1_w2 / count_w1
        else:
            prob = 0.0
            
        sentence_prob *= prob
        print(f"{(w1 + ' ' + w2):<25} {count_w1_w2:<15} {count_w1:<10} {prob:.4f}")

    print(f"\nTotal Sentence Probability (Bigram MLE): {sentence_prob:.10f}")
    
    # Optional: Laplace Smoothing (Add-One)
    print("\n--- Laplace Smoothing (Add-One) ---")
    vocab_size = len(vocab)
    smoothed_prob = 1.0
    print(f"Vocab Size (V): {vocab_size}")
    print(f"{'Bigram':<25} {'Count+1':<15} {'Count(w1)+V':<15} {'P_smooth':<10}")

    for w1, w2 in test_bigrams:
        count_w1_w2 = bigram_counts[(w1, w2)]
        count_w1 = bigram_context_counts[w1]
        
        # P_add1(w2|w1) = (count(w1,w2) + 1) / (count(w1) + V)
        # Note: Depending on definition, V might be the vocab size (unique types)
        
        prob_smooth = (count_w1_w2 + 1) / (count_w1 + vocab_size)
        smoothed_prob *= prob_smooth
        print(f"{(w1 + ' ' + w2):<25} {count_w1_w2+1:<15} {count_w1 + vocab_size:<15} {prob_smooth:.4f}")

    print(f"\nTotal Sentence Probability (Add-One Smoothed): {smoothed_prob:.10f}")

if __name__ == "__main__":
    q1_ngram_model()
