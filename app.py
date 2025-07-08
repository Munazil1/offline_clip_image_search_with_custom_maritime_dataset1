print("✅ Running the UPDATED version of app.py")
from flask import Flask, render_template, request, redirect
import torch, clip, numpy as np, json, os
from utils import (
    get_top_fusion_results_with_clipscore,
    expand_query_words,
    load_feedback_scores,
    load_common_tags
)
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
import webbrowser
import threading
from PIL import Image
from werkzeug.utils import secure_filename

nltk.download('wordnet', quiet=True)

app = Flask(__name__)
device = "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/16", device=device)

# ✅ Load updated compressed features
fusion_features = np.load("fusion_features.npz")["data"]
image_features = np.load("image_features.npz")["data"]

# ✅ Load metadata
with open("caption_image_map.json") as f:
    image_paths = json.load(f)
with open("caption_texts.json") as f:
    display_captions = json.load(f)

# ✅ Load feedback and common tags
feedback_scores = load_feedback_scores("feedback.json")
suggested_tags = load_common_tags("common_tags.json")

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        # ⬆️ Reverse Image Search
        if "upload" in request.files:
            file = request.files["upload"]
            filename = secure_filename(file.filename)
            if filename:
                path = os.path.join("static/uploads", filename)
                file.save(path)

                image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(image)
                    feat /= feat.norm(dim=-1, keepdim=True)
                    feat = feat.cpu().numpy()

                sims = image_features @ feat.T
                top_k = np.argsort(sims.squeeze())[::-1][:5]
                results = [(image_paths[i], display_captions[i], float(sims[i])) for i in top_k]

        # 🔤 Text-based search
        elif "query" in request.form:
            query = request.form["query"].strip().lower().replace(".", "")

            expanded_terms = expand_query_words(query)
            query_vecs = []
            for term in expanded_terms:
                tokens = clip.tokenize([term]).to(device)
                with torch.no_grad():
                    vec = model.encode_text(tokens)
                    vec /= vec.norm(dim=-1, keepdim=True)
                query_vecs.append(vec.cpu().numpy())

            query_feat = np.mean(query_vecs, axis=0)
            query_feat /= np.linalg.norm(query_feat, axis=1, keepdims=True)

            results = get_top_fusion_results_with_clipscore(
                query_feat,
                fusion_features,
                image_features,
                image_paths,
                display_captions,
                feedback_scores,
                model,
                top_k=5
            )

            with open("search_logs.txt", "a") as log:
                for img, cap, score in results:
                    log.write(f"{query} -> {img} ({score:.4f})\n")

    return render_template("index.html", results=results, tags=suggested_tags)

@app.route("/feedback", methods=["POST"])
def feedback():
    fb = request.form.get("feedback")
    if fb:
        value, img = fb.split("_", 1)
        change = 1 if value == "up" else -1
        with open("feedback.json", "a") as f:
            f.write(json.dumps({"image": img, "change": change}) + "\n")
    return redirect("/")

# 🔁 Auto-open browser
def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    threading.Timer(1.0, open_browser).start()
    app.run(debug=False, port=5000)
