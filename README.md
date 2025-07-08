# 🚢 Offline CLIP Image Search with Custom Maritime Dataset

This project implements an offline image search system using OpenAI’s CLIP model, customized for a maritime dataset. It allows users to input a text query and retrieve the most relevant images based on semantic similarity — all without needing an internet connection.

---

## 🔍 Key Features

- CLIP-powered: Uses CLIP (Contrastive Language–Image Pre-training) for cross-modal embedding.
- Custom Maritime Dataset: Focused on ships, cargo, vessels, and related maritime scenes.
- Offline Search: No API calls or cloud services — runs entirely on local hardware.
- Fast Retrieval: Precomputed embeddings ensure quick similarity matching.
- Tag and Caption Support: Optional JSON-based caption maps improve result filtering.

---

## 📁 Project Structure

clip_image_search/
├── app.py                       
├── utils.py                      
├── preprocess_features.py        
├── templates/index.html         
├── caption_image_map.json        
├── requirements.txt              
└── README.md                  
---

## 🚀 How to Run

### 1. Clone the repository

git clone https://github.com/Munazil1/offline_clip_image_search_with_custom_maritime_dataset1.git
cd offline_clip_image_search_with_custom_maritime_dataset1

### 2. Install dependencies

pip install -r requirements.txt

### 3. Download the Maritime Dataset

Download from Google Drive:

https://drive.google.com/drive/folders/1GYeKbdJfk2Eq00AICZgBcP8MZQCr97mV?usp=sharing

After downloading:
- Extract it inside the project folder under a directory called ship_dataset/
- Ensure the images are directly inside the folder (not nested in subfolders)

### 4. Run Preprocessing

python preprocess_features.py

### 5. Start the Application

python app.py

---

## 🧠 How It Works

- Images and captions are embedded using CLIP.
- All embeddings are stored locally (.npy, .json, etc.).
- When a user enters a query, it's embedded and compared to image vectors using cosine similarity.
- The top matches are returned.

---

## 📦 Dependencies

- torch, CLIP, numpy, flask or streamlit, PIL, scikit-learn
- Optional: faiss for faster similarity search

Install them all via:

pip install -r requirements.txt

---

## 📄 License

This project is open-source and free to use under the MIT License.
