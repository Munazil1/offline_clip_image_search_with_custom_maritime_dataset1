import json
from collections import Counter
import re

caption_file = "caption_texts.json"  # or your raw caption list file
output_file = "common_tags.json"

with open(caption_file, "r") as f:
    captions = json.load(f)

# Basic word cleaning
words = []
for cap in captions:
    cap = re.sub(r"[^a-zA-Z0-9 ]", "", cap.lower())
    words.extend(cap.split())

# Exclude stop words (you can extend this list)
stopwords = set(["a", "the", "in", "on", "with", "and", "at", "of", "by", "is", "to", "for"])
filtered_words = [w for w in words if w not in stopwords and len(w) > 2]

# Get top 20 common keywords
top_tags = [word for word, _ in Counter(filtered_words).most_common(20)]

with open(output_file, "w") as f:
    json.dump(top_tags, f)

print("✅ Saved tag suggestions to:", output_file)
