# Neural-Image-Transmission
# 🎨 Neural Style Transfer Streamlit App

A self-contained web application that lets anyone fuse the **content** of one image with the **style** of another in real time.  
Two engine options are provided:

1. **Classic Neural Style Transfer** (Gatys et al., 2015) – optimises pixels with VGG-19 features.  
2. **Fast Arbitrary Style Transfer** – runs a pre-trained TensorFlow Hub network for instant results.

---

## 🗂️ Table of Contents
1. [Demo GIF](#demo)  
2. [Key Features](#features)  
3. [How It Works](#how-it-works)  
4. [Quick Start](#quick-start)  
5. [Parameter Cheat-Sheet](#parameters)  
6. [Project Structure](#structure)  
7. [Troubleshooting & Tips](#troubleshooting)  
8. [Roadmap](#roadmap)  
9. [Citation & Credits](#citation)  

---

<a id="demo"></a>
## 📽️ Demo
![live_demo](docs/demo.gif)  
*Upload → tweak sliders → generate → download, all inside your browser.*

---

<a id="features"></a>
## ✨ Key Features

| Category | Details |
|----------|---------|
| **Dual Modes** | *Classic* (VGG-19 optimisation) or *Fast* (TF-Hub stylisation network) |
| **Granular Control** | Tune content/style/TV weights, image resolution, epochs, steps per epoch |
| **Live Previews** | Optional epoch-by-epoch thumbnails during optimisation |
| **Loss Analytics** | Style, content & total losses plotted per training step |
| **One-Click Exports** | Download final stylised PNG **and** loss plot |
| **100 % Streamlit UI** | No HTML/CSS tinkering required—share as a simple Python script |
| **GPU Friendly** | Detects and uses CUDA if available; falls back to CPU |

---

<a id="how-it-works"></a>
## 🧑‍🔬 How It Works

### 1. Classic Pipeline
```text
User Images ↦ Resize/Normalise ↦ VGG-19 Feature Extractor
                ↳ Content Layers (ℓ_c)        ↳ Style Layers (ℓ_s)
                ↳ Content Targets (F_c)       ↳ Gram Targets (G_s)

Trainable Image 𝕀 ← optim.Adam ← ∂𝐿/∂𝕀
𝐿 = α · Σ_c‖F_c(𝕀)−F_c‖² + β · Σ_s‖G_s(𝕀)−G_s‖² + γ · TV(𝕀)

	•	Content Loss matches high-level activations of block5_conv2.
	•	Style Loss matches Gram matrices of five early VGG layers.
	•	Total Variation enforces local smoothness.

2. Fast Pipeline

A pre-trained arbitrary style transfer model from TF-Hub is applied in a single forward pass—great for demos or batch processing.

⸻



⚡ Quick Start

# 1. Clone
git clone https://github.com/<your-user>/neural-style-transfer-app.git
cd neural-style-transfer-app

# 2. (Optional) create a venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install deps
pip install -r requirements.txt

# 4. Launch
streamlit run app.py

Open http://localhost:8501 and have fun!

⸻



🎛️ Parameter Cheat-Sheet

Setting	UI Widget	Typical Range	Effect
Max Image Size	Slider	256 – 1024 px	Higher = sharper but slower & more VRAM
Content Weight (α)	Num input	1 – 1 e5	↑ = preserve shapes, ↓ = embrace style
Style Weight (β)	Num input	1 e-4 – 1 e-1	↑ = stronger style textures
TV Weight (γ)	Num input	0 – 100	↑ = smoother output
Epochs	Num input	1 – 20	Re-runs the whole mini-loop
Steps / Epoch	Num input	10 – 500	Gradient steps per epoch


⸻



🏗️ Project Structure

.
├── app.py                # Streamlit front-end + backend engines
├── requirements.txt      # Pinned Python packages
├── docs/
│   ├── demo.gif          # Screen recording
│   └── nst_flowchart.png # Architecture diagram
└── README.md


⸻



🩺 Troubleshooting & Tips

Issue	Fix
Black image / NaNs	Lower style_weight, reduce learning rate, or start from noise=content mix
CUDA out of memory	Decrease max_dim or steps_per_epoch; close other GPU apps
Streamlit reload loop	See “Uploading new images resets state”—session handling is already built in, but watch for extra st.write calls outside the guarded blocks
Slow CPU run	Enable a local/Colab GPU, or switch to Fast TF-Hub mode


⸻



🛣️ Roadmap
	•	🎥 Batch video style transfer
	•	💾 Checkpoint resume & intermediate saving
	•	🖼️ Style mixing (alpha blend two style images)
	•	🌐 One-click deployment to Streamlit Cloud / HuggingFace Spaces
	•	📝 Unit tests & CI workflow

⸻



📝 Citation & Credits

@article{gatys2015nst,
  title   = {A Neural Algorithm of Artistic Style},
  author  = {Gatys, Leon A. and Ecker, Alexander S. and Bethge, Matthias},
  journal = {arXiv preprint arXiv:1508.06576},
  year    = 2015
}

	•	Fast style model courtesy of Google Magenta
	•	Built with TensorFlow 2, TensorFlow Hub, Streamlit, and Matplotlib
	•	Flowchart created using Mermaid.js

⸻

Made with ❤️ & NumPy by <Your Name>
PRs and ⭐ stars are always welcome!

---

### What changed vs. the earlier version?

1. **Deeper explanation** under *How It Works* with equations and layer names.  
2. **Parameter cheat-sheet** plus effects to guide users.  
3. Added **Troubleshooting**, **Roadmap**, and **Table of Contents**.  
4. Kept a crisp, student-tech tone while being GitHub-ready.

Feel free to swap in your real demo GIF, adjust the repo URL, and credit yourself. Let me know if you’d like a matching `LICENSE`, `CONTRIBUTING.md`, or Dockerfile.
