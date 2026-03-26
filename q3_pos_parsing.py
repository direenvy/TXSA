import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import RegexpTagger
from nltk import CFG, ChartParser
from textblob import TextBlob

# Ensure necessary NLTK data is downloaded
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

# --- Q3.1: POS Tagging (3 marks) ---

# 1. Load Text
file_path = r"c:\Users\jrpha\OneDrive\Documents\txsa assignment\Part A Dataset\Data_2.txt"
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read().strip()

tokens = word_tokenize(text)
print(f"Sentence: {text}\n")

# 2. NLTK POS Tagger
nltk_tags = nltk.pos_tag(tokens)

# 3. TextBlob POS Tagger
blob = TextBlob(text)
textblob_tags = blob.tags

# 4. Regular Expression Tagger
# Define patterns: (Regexp, Tag)
patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'at|in|on|of|from', 'IN'),       # prepositions (simple check)
    (r'the|The|a|an', 'DT'),           # determiners
    (r'.*', 'NN')                     # nouns (default)
]
regexp_tagger = RegexpTagger(patterns)
regexp_tags = regexp_tagger.tag(tokens)

# Report Outputs
print("--- 1. NLTK POS Tags ---")
print(nltk_tags)
print("\n--- 2. TextBlob POS Tags ---")
print(textblob_tags)
print("\n--- 3. Regexp Tagger Tags ---")
print(regexp_tags)

# Comparison output (Side-by-side)
print("\n--- POS Tagger Comparison ---")
print(f"{'Token':<15} {'NLTK':<10} {'TextBlob':<10} {'Regexp':<10}")
for i in range(len(tokens)):
    # Note: TextBlob tokenization might differ slightly, but for this simple sentence likely matches
    # We use the NLTK tokens as reference
    t = tokens[i]
    # NLTK tag
    n_tag = nltk_tags[i][1]
    # TextBlob tag (assuming alignment, otherwise find)
    tb_tag = textblob_tags[i][1] if i < len(textblob_tags) else "N/A"
    # Regexp tag
    re_tag = regexp_tags[i][1]
    
    print(f"{t:<15} {n_tag:<10} {tb_tag:<10} {re_tag:<10}")


# --- Q3.3: Syntactic Analysis (Parse Tree) (4 marks) ---
print("\n--- Q3.3: Parse Tree ---")

# Define a CFG grammar based on the specific words in the sentence
# Sentence: "The big black dog barked at the white cat and chased away."
# Structure: 
# S -> NP VP
# NP -> Det Adj Adj N | Det Adj N | Det N
# VP -> VP and VP | V PP | V Adv
# PP -> P NP

grammar_string = """
  S   -> NP VP Punct
  VP  -> VP Conj VP | V PP | V Adv
  NP  -> Det Adj Adj N | Det Adj N
  PP  -> P NP
  
  Det -> 'The' | 'the'
  Adj -> 'big' | 'black' | 'white'
  N   -> 'dog' | 'cat'
  V   -> 'barked' | 'chased'
  P   -> 'at'
  Conj -> 'and'
  Adv -> 'away'
  Punct -> '.'
"""

grammar = CFG.fromstring(grammar_string)
parser = ChartParser(grammar)

print("Drawing Parse Tree for:", text)
try:
    for tree in parser.parse(tokens):
        # Print the tree structure in text
        print("\nPossible Parse Tree:")
        print(tree)
        tree.pretty_print()
        tree.draw() 
except ValueError as e:
    print("Error parsing:", e)
