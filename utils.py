import numpy as np
import os, json
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet
from collections import defaultdict

# Ensure WordNet is available
nltk.download('wordnet', quiet=True)

def expand_query_words(query):
    """
    Expands a user query with synonyms using WordNet.
    """
    words = query.lower().split()
    expanded = set(words)
    for word in words:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    return list(expanded)

def load_feedback_scores(path="feedback.json"):
    """
    Loads feedback from a local JSONL file and returns cumulative scores per image.
    """
    scores = defaultdict(int)
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    scores[entry["image"]] += entry["change"]
                except:
                    continue
    return scores

def load_common_tags(tag_file="common_tags.json"):
    """
    Loads a list of common tags from a JSON file.
    """
    if os.path.exists(tag_file):
        with open(tag_file) as f:
            return json.load(f)
    return []

def get_top_fusion_results_with_clipscore(
    query_features,
    fusion_features,
    image_features,
    image_paths,
    captions,
    feedback_scores,
    model,  # Optional: pass clip model if needed later
    top_k=5,
    min_image_distance=0.15
):
    """
    Returns top-k results using:
    - fusion vector similarity
    - feedback-based reweighting
    - optional personalized adaptation to positive feedback
    """
    sims = fusion_features @ query_features.T
    scores = sims.squeeze()

    # Store selected results
    scored = []
    selected_feats = []
    seen = set()

    for i in np.argsort(scores)[::-1]:
        img = image_paths[i]
        cap = captions[i]
        image_feat = image_features[i]
        score = float(scores[i])

        # Feedback weight
        feedback = feedback_scores.get(img, 0)
        score += 0.1 * feedback

        # OPTIONAL: Improve visual alignment if liked
        if feedback > 0:
            # Move the image vector slightly toward query vector
            image_feat = image_feat + 0.2 * query_features.squeeze()

        # Visual diversity filtering
        if img in seen:
            continue
        if not selected_feats:
            selected_feats.append(image_feat)
            scored.append((img, cap, round(score, 4)))
            seen.add(img)
        else:
            max_sim = cosine_similarity([image_feat], selected_feats).max()
            if max_sim < (1 - min_image_distance):
                selected_feats.append(image_feat)
                scored.append((img, cap, round(score, 4)))
                seen.add(img)

        if len(scored) == top_k:
            break

    return scored
