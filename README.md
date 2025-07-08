# offline_clip_image_search_with_custom_maritime_dataset
A fully offline semantic image search tool using OpenAI's CLIP ViT-L/14 model and a custom maritime dataset. Supports both natural language and reverse image queries, with a feedback loop and Flask-based desktop UI. Designed for secure, air-gapped environments like defense and surveillance systems.

clip_image_search/
├── app.py # Main Flask/Streamlit app
├── utils.py # Utility functions for embedding and search
├── preprocess_features.py # Preprocessing & embedding images
├── templates/index.html # Web UI (if Flask used)
├── caption_image_map.json # Optional: image to caption mapping
├── requirements.txt # Python dependencies
├── .gitignore # Files to exclude from Git
└── README.md # Project documentation

bash
Copy
Edit

## 🚀 How to Run

1. **Clone the repository**

git clone https://github.com/Munazil1/offline_clip_image_search_with_custom_maritime_dataset1.git
cd offline_clip_image_search_with_custom_maritime_dataset1
Install dependencies


pip install -r requirements.txt
Run the app

python app.py
Or use Streamlit/Gradio if configured.

🧠 How It Works
Images and captions are embedded using CLIP.

All embeddings are stored locally (.npy, .json, etc.).

When a query is entered, it’s embedded and compared to image vectors using cosine similarity.

The top matches are returned and displayed.

📦 Dependencies
openai-clip or transformers

torch

flask, streamlit, or gradio

numpy, PIL, scikit-learn, faiss (optional)

Install them using:

pip install -r requirements.txt
📸 Sample Use Case
Search for:

"Red cargo ship in storm"

And get back:

cargo_red_002.jpg

stormy_ocean_tanker.jpg

📄 License
This project is open-source and free to use under the MIT License.
