import torch, clip, numpy as np
import os, json
from PIL import Image
from tqdm import tqdm

device = "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

caption_file = "ship_dataset.txt"
image_folder = "static/images/"

captions = []
image_paths = []
caption_features = []
image_features = []
fusion_features = []

image_cache = {}

with open(caption_file, "r", encoding="utf-8") as f:
    for line in f:
        if ":" not in line:
            continue
        img, caption = line.strip().split(":", 1)
        img = img.strip()
        caption = caption.strip().lower().replace(".", "")  # ✅ clean caption
        image_path = os.path.join(image_folder, img)

        if not os.path.exists(image_path):
            print(f"❌ Missing image: {img}")
            continue

        # Text feature
        tokens = clip.tokenize([caption]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)

        # Image feature (cached)
        if img in image_cache:
            image_feat = image_cache[img]
        else:
            try:
                image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_feat = model.encode_image(image)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_cache[img] = image_feat
            except:
                print(f"⚠️ Skipping broken image: {img}")
                continue

        # Fusion feature (more weight to text)
        fusion_feat = 0.9 * text_feat + 0.1 * image_feat
        fusion_feat /= fusion_feat.norm(dim=-1, keepdim=True)

        # Save data
        captions.append(caption)
        image_paths.append(img)
        caption_features.append(text_feat.cpu().numpy())
        image_features.append(image_feat.cpu().numpy())
        fusion_features.append(fusion_feat.cpu().numpy())

# ✅ Save compressed numpy files
np.savez_compressed("caption_features.npz", data=np.concatenate(caption_features))
np.savez_compressed("image_features.npz", data=np.concatenate(image_features))
np.savez_compressed("fusion_features.npz", data=np.concatenate(fusion_features))

# ✅ Save metadata
with open("caption_texts.json", "w") as f:
    json.dump(captions, f)
with open("caption_image_map.json", "w") as f:
    json.dump(image_paths, f)

print(f"✅ Done: Saved {len(captions)} caption-specific vectors.")
