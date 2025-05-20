# 🎨 Neural Style Transfer Web App

Hey there! 👋 This is a simple and interactive **Neural Style Transfer** app built with **Streamlit** and **TensorFlow**. You can upload two images — one as content and one as style — and the app will blend them to generate a brand-new, stylized image. You can also visualize training loss and download both the output and the loss plot.

There are two modes you can choose from:
- **Classic Neural Style Transfer** using VGG19 (slower but gives you more control),
- **Fast Style Transfer** using a pre-trained model from TensorFlow Hub (super quick and great for instant results).

---

## 📸 Demo

https://drive.google.com/file/d/1h9Plpw7fFEt8va81M9YWNSGYm7nMSJFV/view?usp=sharing

---

## 📽️ Presentation

https://drive.google.com/file/d/16Hi0Ry0cELMLyfClyD7ykkyeCxdMu-zp/view?usp=sharing

---
## 🔍 What This App Does

- Lets you **upload content and style images**
- Runs **neural style transfer** to merge them
- You can **adjust parameters** like content/style weights, resolution, etc.
- Shows you **live previews** if you want
- **Plots the losses** (style, content, total)
- You can **download the final image** and the **loss plot**

---

## 🧠 How It Works

### 1. Classic Mode (based on Gatys et al.)

We use the VGG19 model (without the top layers) and extract features from:
- **Content layer:** `block5_conv2`
- **Style layers:** `block1_conv1`, ..., `block5_conv1`

We then define a total loss:

Total Loss = (Content Loss * weight) + (Style Loss * weight) + Total Variation Loss

The image itself is optimized (not the model), so we iteratively update it to minimize the total loss.

### 2. Fast Mode (TF Hub)

We load a pre-trained model from TensorFlow Hub and pass the content + style image to get the result immediately — no training loop required!

# 🎨 Neural Style Transfer App

This is a simple and interactive web application that lets you apply artistic style transfer to your images using deep learning. Built with Streamlit, it allows you to upload your own images and stylize them using pre-trained models.

---

## 🛠️ How to Run the App

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/neural-style-transfer-app.git
cd neural-style-transfer-app
```
### 2. (Optional) Set up a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
streamlit run app.py
```
### Then open your browser and go to:
### 👉 http://localhost:8501

⸻

### 5. 🎛️ Parameters You Can Tune

Parameter	Description
Max Image Size	Controls resolution (higher = better quality but slower)
Content Weight	Controls how much of the content image structure is preserved
Style Weight	Controls how much of the style is transferred
TV Weight	Smooths the output image by penalizing noise
Epochs / Steps	More steps usually lead to better results (classic mode only)
Show Progress	Preview results after each epoch


⸻

### 6. 📁 Project Structure
<pre>
├── app.py                 # Main app script
├── requirements.txt       # Python dependencies
├── docs/
│   └── nst_flowchart.png  # Visual workflow of the app
└── README.md              # This file!
</pre>


⸻

## 🐞 Common Issues

- **Output looks noisy**  
  👉 Try lowering the **Style Weight** or increasing the **TV Weight**.

- **App crashes on large images**  
  👉 Reduce the **Max Image Size** (e.g., set it to `512`).

- **Runs too slowly**  
  👉 Use **Fast Mode**, or reduce the number of **Epochs** and **Steps**.
