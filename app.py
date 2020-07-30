import streamlit as st
from PIL import Image
from style_transfer.solver import transfer_style

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Style Transfer Using PyTorch")

# variables
model_layers = ["conv_%d_%d" % (i, j) for i in range(1, 3) for j in range(1, 3)] + \
    ["conv_%d_%d" % (i, j) for i in range(3, 6) for j in range(1, 5)]

# sidebar contents
st.sidebar.title("Configuration")
content_weight = st.sidebar.number_input("Content weight", value=1.0)
style_weight = st.sidebar.number_input("Style weight", value=1000000.0)
image_size = st.sidebar.selectbox(
    "Output image width",
    (64, 128, 256, 512),
    index=3
)
epochs = st.sidebar.slider("Epochs", value=26, min_value=10, max_value=40)

content_layers = st.sidebar.multiselect(
    "Content layers", 
    model_layers, 
    ["conv_4_2"]
)
style_layers = st.sidebar.multiselect(
    "Style layers",
    model_layers,
    ["conv_%d_1" % i for i in range(1, 6)]
)
style_image = st.sidebar.file_uploader("Choose style image", type="jpg")
style_image_placeholder = st.sidebar.empty()
if style_image is not None:
    style_image_placeholder.image(style_image, use_column_width=True)

content_image = st.sidebar.file_uploader("Choose content image", type="jpg")
content_image_placeholder = st.sidebar.empty()
if content_image is not None:
    content_image_placeholder.image(content_image, use_column_width=True)
st.sidebar.text('')
button = st.sidebar.button("Transfer!")

# page contents
with open("readme.md", "r") as f:
    content = f.readlines()
    content = content[2:]
    content[0] = "[This repository](https://github.com/firekind/style-transfer) contains an implementation of [Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)."
    content[1] += " Scroll down to try it out!\n"
    st.markdown("".join(content), unsafe_allow_html=True)

st.markdown("## Try it Out!")
if button:
    if style_image == None or content_image == None:
        st.error("Style and/or content images are not selected")
    elif len(content_layers) == 0 or len(style_layers) == 0:
        st.error("Please select at least one style and content layer")
    else:
        header = st.empty()
        result = st.empty()

        header.markdown("Transferring style")
        prog_bar = result.progress(0)
        current_progress = [0]

        def post_epoch_callback():
            current_progress[0] += 1
            prog_bar.progress(current_progress[0] / epochs)

        output = transfer_style(
            style_image=Image.open(style_image),
            content_image=Image.open(content_image),
            image_size=image_size,
            epochs=epochs,
            content_weight=content_weight,
            style_weight=style_weight,
            content_layers=content_layers,
            style_layers=style_layers,
            verbose=False,
            post_epoch_callback=post_epoch_callback,
        )
        header.markdown("Transfer Complete! Here's the result:")
        result.image(output)
else:
    st.info("Select the images using the sidebar and click transfer!")