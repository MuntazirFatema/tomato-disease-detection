# 🍅 Tomato Disease Detection

A deep learning-based web app that detects diseases in tomato leaves from uploaded images using a Convolutional Neural Network (CNN), built with **TensorFlow/Keras** and deployed with **Streamlit**.

🔗 **Live App:** [tomato-disease-detection.streamlit.app](https://tomato-disease-detection-nthzcywz6vq95uuux5ntcm.streamlit.app)

---

## 📌 About the Project

This project uses a CNN trained on the PlantVillage dataset to classify tomato leaf images into 10 categories — 9 diseases plus a healthy class. Users can upload a photo of a tomato leaf, and the model predicts the disease along with a confidence score.

The goal is to help farmers, students, and agricultural researchers quickly identify tomato leaf diseases without needing expert diagnosis, making early detection more accessible.

---

## 🧠 Model Details

- **Framework:** TensorFlow / Keras
- **Architecture:** Convolutional Neural Network (CNN)
- **Input size:** 128x128 RGB images
- **Classes predicted (10):**
  - Tomato Bacterial Spot
  - Tomato Early Blight
  - Tomato Late Blight
  - Tomato Leaf Mold
  - Tomato Septoria Leaf Spot
  - Tomato Spider Mites (Two-Spotted Spider Mite)
  - Tomato Target Spot
  - Tomato Yellow Leaf Curl Virus
  - Tomato Mosaic Virus
  - Tomato Healthy

The app also applies a confidence threshold (75%) — if the model isn't confident enough, it flags the result as "Unrecognized" rather than forcing a guess.

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras (CNN)
- **Frontend & Serving:** Streamlit
- **Image Processing:** Pillow, NumPy
- **Deployment:** Streamlit Community Cloud

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/MuntazirFatema/tomato-disease-detection.git
cd tomato-disease-detection

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ⚠️ Known Limitation

This model was trained **only on tomato leaf images**, so it does not currently know what a "non-leaf" object looks like. If you upload a photo of something other than a tomato leaf (e.g. a person, an object, or another plant), the model may still confidently predict one of the 10 disease classes, since it has never seen a "not a leaf" example during training.

This will be addressed in a future update (see below).

---

## 🔮 Planned Improvements

- [ ] Add a "non-leaf" / "unknown object" class by training on a diverse set of non-leaf images, so the model can properly reject irrelevant uploads instead of forcing a wrong prediction
- [ ] Expand the model to detect diseases in **potato** and **pepper (bell pepper)** leaves as well, combining all three crops into a single, more comprehensive plant disease detection app
- [ ] Improve UI/UX with a more polished, farmer-friendly interface
- [ ] Add multi-language support (Hindi/Urdu) to make the app more accessible to farmers
- [ ] Add treatment/remedy suggestions alongside each disease prediction

The long-term goal is to grow this into a complete, easy-to-use plant disease detection tool that can genuinely help farmers identify and treat crop diseases early — starting with tomato, and expanding to other common crops over time.

---



## 🤝 Contributing

This is a personal learning project, but suggestions and feedback are always welcome. Feel free to open an issue or reach out.

---

## 📄 License

This project is open for educational and research purposes.

---

## 👤 Author

**Muntazir Fatema**
Built as part of a computer vision learning journey — from model training to full deployment.
