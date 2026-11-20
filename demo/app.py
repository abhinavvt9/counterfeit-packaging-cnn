"""
Demonstration app: upload a packaging photo -> genuine / counterfeit + Grad-CAM.

Run:   python demo/app.py            (opens in the browser, http://127.0.0.1:7860)
In Colab the same command prints a public link.

The image goes through EXACTLY the same preprocessing as the training images
(controlled protocol). The number shown is the model's score for the
counterfeit class. It is not a calibrated probability.
"""
import json
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from cnn.gradcam import gradcam, overlay  # noqa: E402
from cnn.model import build_model  # noqa: E402
from preprocessing.preprocessing import preprocess_array  # noqa: E402

META = json.loads((config.MODELS_DIR / "final_config.json").read_text())
DEVICE = torch.device("cpu")
MODEL = build_model(META["config"], img_size=META["img_size"])
MODEL.load_state_dict(torch.load(config.MODELS_DIR / "final_cnn.pt", map_location=DEVICE))
MODEL.eval()

DISCLAIMER = ("Research demonstration only. The model classifies packaging images based on "
              "patterns in one small public dataset, whose two classes were photographed "
              "differently. It cannot verify a medicine. Real authentication needs laboratory "
              "or regulatory checks.")


def analyse(image):
    if image is None:
        return "Please upload an image.", None, None
    rgb = np.asarray(image.convert("RGB"))
    x = preprocess_array(rgb, "controlled", META["img_size"])
    cam, score = gradcam(MODEL, x, DEVICE)
    label = "COUNTERFEIT" if score >= 0.5 else "GENUINE"
    text = (f"## Prediction: {label}\n"
            f"Model score for 'counterfeit': **{score:.2f}** (threshold 0.50)\n\n"
            f"Model: {META['experiment']}, {META['n_parameters']:,} parameters, trained from scratch.")
    return text, x, overlay(x, cam)


with gr.Blocks(title="Counterfeit Packaging Screening") as demo:
    gr.Markdown("# Counterfeit Pharmaceutical Packaging Screening\n" + DISCLAIMER)
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload packaging image")
        with gr.Column():
            result = gr.Markdown()
            with gr.Row():
                seen = gr.Image(label="What the model sees (preprocessed 128x128)")
                heat = gr.Image(label="Grad-CAM (red = evidence for counterfeit)")
    gr.Button("Analyze", variant="primary").click(analyse, inp, [result, seen, heat])

if __name__ == "__main__":
    import os
    demo.launch(share="COLAB_RELEASE_TAG" in os.environ)   # public link only inside Colab
