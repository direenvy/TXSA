import re
from nltk.stem import RegexpStemmer, PorterStemmer, LancasterStemmer

text = open(r"c:\Users\jrpha\OneDrive\Documents\txsa assignment\Part A Dataset\Data_1.txt", encoding="utf-8").read()
tokens = re.findall(r"[A-Za-z']+", text.lower())

rs = RegexpStemmer("ing$|s$|e$|able$", min=4)
ps = PorterStemmer()
ls = LancasterStemmer()

print("TOKENS", len(tokens))
print("FIRST40", tokens[:40])
print("REGEXP_FIRST40", [rs.stem(t) for t in tokens[:40]])
print("PORTER_FIRST40", [ps.stem(t) for t in tokens[:40]])
print("LANCASTER_FIRST40", [ls.stem(t) for t in tokens[:40]])

words = [
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

print("WORD_COMPARISON")
for w in words:
    print(w, rs.stem(w), ps.stem(w), ls.stem(w))
