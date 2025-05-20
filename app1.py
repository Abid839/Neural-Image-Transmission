import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import optimizers
from tensorflow.keras.applications import vgg19
from PIL import Image
import io
import time
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Neural Style Transfer", layout="wide")
st.title("🎨 Neural Style Transfer Web App")

# Sidebar for settings
st.sidebar.header("Manual Parameters")
max_dim = st.sidebar.slider("Max Image Size (px)", 256, 1024, 512, 64)
content_weight = st.sidebar.number_input("Content Weight", min_value=1.0, max_value=1e5, value=1e4, step=1.0)
style_weight = st.sidebar.number_input("Style Weight", min_value=0.0, max_value=1e2, value=1e-2, step=0.01, format="%.4f")
tv_weight = st.sidebar.number_input("Total Variation Weight", min_value=0.0, max_value=1e2, value=30.0, step=0.1, format="%.2f")
epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=20, value=5, step=1)
steps_per_epoch = st.sidebar.number_input("Steps per Epoch", min_value=10, max_value=500, value=100, step=10)
show_progress = st.sidebar.checkbox("Show Progress Images", value=True)
st.sidebar.header("Auto Parameters")
use_fast = st.sidebar.checkbox("Use Fast Style Transfer (TF Hub)", value=False)

# Helper functions
def load_img(file, max_dim=512):
    img = Image.open(file).convert('RGB')
    # Resize while keeping aspect ratio
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    img = np.array(img).astype(np.float32) / 255.0
    img = tf.convert_to_tensor(img)
    img = tf.expand_dims(img, 0)
    return img

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def vgg_layers(layer_names):
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}

def style_transfer(content_image, style_image,
                   content_weight=1e4, style_weight=1e-2, total_variation_weight=30,
                   epochs=5, steps_per_epoch=100, max_dim=512, show_progress=True, st_container=None):

    # Define layers
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    # Create model
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # Initialize image
    image = tf.Variable(content_image)
    opt = optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            outputs = extractor(image)
            style_outputs = outputs['style']
            content_outputs = outputs['content']
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                                   for name in style_outputs.keys()])
            style_loss *= style_weight / len(style_layers)
            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                                     for name in content_outputs.keys()])
            content_loss *= content_weight / len(content_layers)
            loss = style_loss + content_loss
            loss += total_variation_weight * tf.image.total_variation(image)
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, 0.0, 1.0))
        return style_loss, content_loss, loss

    all_losses = []
    start = time.time()
    progress_images = []
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            style_loss, content_loss, loss = train_step(image)
            # Collect losses for every step
            all_losses.append([
                float(np.squeeze(style_loss.numpy())),
                float(np.squeeze(content_loss.numpy())),
                float(np.squeeze(loss.numpy()))
            ])
        # Show progress
        if show_progress and st_container:
            st_container.image(tensor_to_image(image), caption=f"Epoch {epoch+1}", use_container_width=True)
    end = time.time()
    return image, all_losses, end - start

# --- Streamlit main workflow ---

# Upload images
col1, col2 = st.columns(2)
with col1:
    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"], key="content")
with col2:
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"], key="style")

# Clear session state when new images are uploaded or parameters change
def reset_session_state():
    for key in ['stylized_result', 'stylized_result_hub']:
        if key in st.session_state:
            del st.session_state[key]

if 'prev_content_file' not in st.session_state:
    st.session_state['prev_content_file'] = None
if 'prev_style_file' not in st.session_state:
    st.session_state['prev_style_file'] = None
if 'prev_params' not in st.session_state:
    st.session_state['prev_params'] = None

current_params = (max_dim, content_weight, style_weight, tv_weight, epochs, steps_per_epoch, show_progress, use_fast)

if content_file != st.session_state['prev_content_file'] or style_file != st.session_state['prev_style_file'] or current_params != st.session_state['prev_params']:
    reset_session_state()
    st.session_state['prev_content_file'] = content_file
    st.session_state['prev_style_file'] = style_file
    st.session_state['prev_params'] = current_params

if content_file and style_file:
    # Show uploaded images for both classic and TF Hub modes
    st.write("## 🖼️ Uploaded Images")
    img_col1, img_col2 = st.columns(2)
    # Load images for display (always as PIL)
    display_content_img = Image.open(content_file).convert('RGB')
    display_style_img = Image.open(style_file).convert('RGB')
    img_col1.image(display_content_img, caption="Content Image", use_container_width=True)
    img_col2.image(display_style_img, caption="Style Image", use_container_width=True)
    st.markdown("---")

    # Tabs for UI
    tabs = st.tabs(["🎯 Input", "🎆 Stylized Output", "📈 Loss Plot"])

    with tabs[0]:
        st.markdown("<h2 style='font-size:2em; font-weight:bold;'>🎯 Input</h2>", unsafe_allow_html=True)
        st.write("### Input Images")
        col_a, col_b = st.columns(2)
        col_a.image(display_content_img, caption="Content Image", use_container_width=True)
        col_b.image(display_style_img, caption="Style Image", use_container_width=True)

    # Main NST logic
    if not use_fast:
        content_img = load_img(content_file, max_dim=max_dim)
        style_img = load_img(style_file, max_dim=max_dim)
        with tabs[1]:
            st.markdown("<h2 style='font-size:2em; font-weight:bold;'>🎆 Stylized Output</h2>", unsafe_allow_html=True)
            # Button to run style transfer
            if st.button("✨ Generate Stylized Image", key="classic_btn"):
                with st.spinner("Running Neural Style Transfer... 🎨 This may take a minute."):
                    progress_area = st.empty()
                    start_time = time.time()
                    stylized_img, all_losses, run_time = style_transfer(
                        content_img, style_img,
                        content_weight, style_weight, tv_weight,
                        epochs, steps_per_epoch,
                        max_dim=max_dim,
                        show_progress=show_progress,
                        st_container=progress_area if show_progress else None
                    )
                    elapsed = time.time() - start_time
                st.session_state['stylized_result'] = {
                    'image': stylized_img,
                    'all_losses': all_losses,
                    'run_time': run_time
                }
            if 'stylized_result' in st.session_state:
                stylized_img = st.session_state['stylized_result']['image']
                all_losses = st.session_state['stylized_result']['all_losses']
                run_time = st.session_state['stylized_result']['run_time']
                st.image(tensor_to_image(stylized_img), caption="Stylized Output", use_container_width=True)
                st.success(
                    f"✅ **Done!**\n\n"
                    f"- **Epochs:** {epochs}\n"
                    f"- **Steps per Epoch:** {steps_per_epoch}\n"
                    f"- **Elapsed Time:** {run_time:.1f} seconds"
                )
                # Display losses
                st.markdown("### Loss Value")
                final_style_loss = all_losses[-1][0]
                final_content_loss = all_losses[-1][1]
                final_total_loss = all_losses[-1][2]
                st.write(f"- Total Loss: {float(final_total_loss):.4f}")
                st.write(f"- Style Loss: {float(final_style_loss):.4f}")
                st.write(f"- Content Loss: {float(final_content_loss):.4f}")
                buf = io.BytesIO()
                tensor_to_image(stylized_img).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Stylized Image",
                    data=buf.getvalue(),
                    file_name="stylized.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_classic"
                )
                # st.balloons()
                # Show loss curves in Loss Plot tab
                if all_losses:
                    with tabs[2]:
                        st.markdown("<h2 style='font-size:2em; font-weight:bold;'>📈 Loss Plot</h2>", unsafe_allow_html=True)
                        import matplotlib.pyplot as plt
                        all_losses_np = np.array(all_losses)
                        fig = plt.figure(figsize=(8, 4))
                        steps_list = list(range(1, len(all_losses_np) + 1))
                        plt.plot(steps_list, all_losses_np[:,0], label="Style Loss")
                        plt.plot(steps_list, all_losses_np[:,1], label="Content Loss")
                        plt.plot(steps_list, all_losses_np[:,2], label="Total Loss")
                        plt.xlabel("Training Step")
                        plt.legend()
                        plt.title("Training Losses")
                        st.pyplot(fig)
                        plot_buf = io.BytesIO()
                        fig.savefig(plot_buf, format='png')
                        st.download_button(
                            "⬇️ Download Loss Plot",
                            data=plot_buf.getvalue(),
                            file_name="loss_plot.png",
                            mime="image/png",
                            use_container_width=True,
                            key="download_loss_plot"
                        )
                        plt.close(fig)
            else:
                st.info("Click **Generate Stylized Image** to begin.")
        # Loss plot: show placeholder if not run yet
        with tabs[2]:
            st.markdown("<h2 style='font-size:2em; font-weight:bold;'>📈 Loss Plot</h2>", unsafe_allow_html=True)
            if 'stylized_result' not in st.session_state:
                st.info("Run style transfer to see the loss plot.")
    else:
        def load_img_hub(file, max_dim=512):
            img = Image.open(file).convert('RGB')
            img = img.resize((max_dim, max_dim))
            img = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
            return img

        content_img_hub = load_img_hub(content_file, max_dim=max_dim)
        style_img_hub = load_img_hub(style_file, max_dim=max_dim)
        with tabs[1]:
            st.markdown("<h2 style='font-size:2em; font-weight:bold;'>🎆 Stylized Output</h2>", unsafe_allow_html=True)
            if st.button("⚡ Generate Stylized Image (TF Hub)", key="hub_btn"):
                with st.spinner("Running Fast Neural Style Transfer (TF Hub)... 🚀"):
                    start_time = time.time()
                    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
                    stylized_image = hub_model(tf.constant(content_img_hub), tf.constant(style_img_hub))[0]
                    stylized_image = np.squeeze(stylized_image.numpy(), axis=0)
                    stylized_image = (stylized_image * 255).astype(np.uint8)
                    elapsed = time.time() - start_time
                st.session_state['stylized_result_hub'] = {
                    'image': stylized_image,
                    'run_time': elapsed
                }
            if 'stylized_result_hub' in st.session_state:
                stylized_image = st.session_state['stylized_result_hub']['image']
                elapsed = st.session_state['stylized_result_hub']['run_time']
                st.image(stylized_image, caption='Stylized Output (TF Hub)', use_container_width=True)
                buf = io.BytesIO()
                Image.fromarray(stylized_image).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download Stylized Image",
                    data=buf.getvalue(),
                    file_name="stylized.png",
                    mime="image/png",
                    use_container_width=True,
                    key="download_hub"
                )
                # st.balloons()
            else:
                st.info("Click **Generate Stylized Image (TF Hub)** to begin.")
        with tabs[2]:
            st.markdown("<h2 style='font-size:2em; font-weight:bold;'>📈 Loss Plot</h2>", unsafe_allow_html=True)
            st.info("Loss plot is only available for classic NST mode.")
else:
    st.info("Upload both content and style images to begin.")