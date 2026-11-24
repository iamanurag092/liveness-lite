"""Build report.pdf (and docs/figures/architecture.png) in plain language.

    pip install reportlab matplotlib
    python scripts/build_report.py [--out report.pdf] [--demo-url https://...]

Numbers come from runs/*/metrics_fp32.json and models/*/model_card.json (see docs/03_experiments.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

DEMO_URL = "https://liveness-lite-demo.vercel.app/"
AUTHOR = "Anurag Tiwari"
REPO = "github.com/iamanurag092/liveness-lite"


def make_architecture_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 1.9))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1. Input", "colour image,\nresized to\n224 x 224", "#e6fcf5", "#0ca678"),
        ("2. Backbone", "MobileNetV3:\nconvolution layers\nfind patterns", "#f3f0ff", "#7048e8"),
        (
            "3. Summary",
            "patterns are\ncondensed into a\nshort list of numbers",
            "#f3f0ff",
            "#7048e8",
        ),
        (
            "4. Classifier",
            "one small layer\ngives a single score\nfrom 0 to 1",
            "#f3f0ff",
            "#7048e8",
        ),
        ("5. Decision", "score above the\ncut-off: SPOOF,\notherwise LIVE", "#e6fcf5", "#0ca678"),
    ]
    width, gap = 0.172, 0.035
    for i, (title, body, face, edge) in enumerate(stages):
        x = i * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.1),
                width,
                0.8,
                boxstyle="round,pad=0.005,rounding_size=0.02",
                facecolor=face,
                edgecolor=edge,
                linewidth=1,
            )
        )
        ax.text(x + width / 2, 0.76, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + width / 2, 0.42, body, ha="center", va="center", fontsize=6.8, linespacing=1.4)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap - 0.004, 0.5),
                xytext=(x + width + 0.004, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
            )
    fig.tight_layout(pad=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build(out: Path, demo_url: str) -> None:
    figure = Path("docs/figures/architecture.png")
    make_architecture_figure(figure)

    base = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=9.6,
        leading=13.2,
        alignment=TA_LEFT,
        spaceAfter=4.5,
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#555555")
    )
    h1 = ParagraphStyle(
        "h1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        spaceAfter=2,
        leading=22,
    )
    h2 = ParagraphStyle(
        "h2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        spaceBefore=9,
        spaceAfter=3,
        textColor=colors.HexColor("#243a8f"),
    )
    sub = ParagraphStyle("sub", parent=body, fontSize=10, textColor=colors.HexColor("#555555"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)

    def P(text, style=body):
        return Paragraph(text, style)

    def B(text):
        return Paragraph(text, bullet, bulletText="-")

    link = f'<link href="{demo_url}" color="#243a8f">{demo_url}</link>'
    head = ParagraphStyle(
        "head", parent=body, fontName="Helvetica-Bold", fontSize=8, leading=9.5, spaceAfter=0
    )
    results = Table(
        [
            [
                Paragraph(t, head)
                for t in (
                    "Model",
                    "Accuracy",
                    "ROC-AUC",
                    "Live images wrongly flagged",
                    "Spoofs wrongly accepted",
                    "File size",
                    "Speed",
                )
            ],
            ["Lite", "96.0%", "0.994", "4.6%", "3.9%", "6 MB", "about 2 ms"],
            ["Accurate", "97.0%", "0.997", "3.1%", "3.0%", "17 MB", "about 6 ms"],
        ],
        colWidths=[2.0 * cm, 1.9 * cm, 1.8 * cm, 3.4 * cm, 3.2 * cm, 1.9 * cm, 2.4 * cm],
    )
    results.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    story = [
        P("Lightweight genuine-vs-spoof image classification", h1),
        P(
            f"Take-home task for Deep Algorithms Solutions &nbsp;|&nbsp; {AUTHOR} &nbsp;|&nbsp; 23 November 2026",
            sub,
        ),
        P(f"Live demo: {link} &nbsp;|&nbsp; Source code: {REPO} (also in the ZIP)", small),
        P("1. The task and what I built", h2),
        P(
            "The task was to build a lightweight deep learning model that classifies an input image as genuine "
            "or spoofed, with high accuracy and low computational cost and memory. A spoof here means a "
            "presentation attack: someone holds up a printed photo, or a screen showing a face, to fool a "
            "face-based system. I built two models of different sizes, measured them, and published a browser "
            "demo where anyone can upload a cropped face image and see the result."
        ),
        P("2. Why this dataset", h2),
        P(
            "I used <b>LCC-FASD</b>, a public face anti-spoofing dataset (Kaggle copy). It has 18,827 face "
            "images, each labelled live or spoof, collected with many different phones. I chose it because:"
        ),
        B("the task is about images, and this dataset is made of images;"),
        B(
            "it has both classes and official lists that separate training images from test images, so my "
            "results are measured the standard way and I did not have to invent my own test;"
        ),
        B("it is small enough to train on a laptop;"),
        B("it can be downloaded immediately, with no licence request."),
        P(
            "Other well-known datasets were not practical here: CASIA-FASD, Replay-Attack and OULU-NPU are "
            "videos and need a licence request; CASIA-SURF is far larger than this task needs; free sample sets "
            "from commercial vendors are too small to train on."
        ),
        P(
            "<b>How I split the data.</b> Training: 12,194 images, used to teach the model. Validation: "
            "2,867 images, taken out of the training lists to choose settings (keeping each person in only one "
            "group). Test: 3,766 images, the official test list, used only once at the end. About 90% of all "
            "images are spoofs, which affects how the results must be read (section 4)."
        ),
        P("3. The model", h2),
        P(
            "<b>The base of the model.</b> I did not train a network from scratch. I started from "
            "<b>MobileNetV3</b>, a small convolutional neural network (CNN) that was already trained on "
            "ImageNet, a large collection of everyday photos. That network already knows how to pick out "
            "edges, textures and shapes. I replaced its last layer with a single output and trained it further "
            "on the face images (fine-tuning, a form of transfer learning). With only about 12,000 training "
            "images this works far better than starting from nothing, and it needs much less computing."
        ),
        P("<b>Architecture.</b> One image goes through five steps:", body),
        Image(str(figure), width=16.4 * cm, height=16.4 * cm * 1.9 / 6.8),
        P("Figure 1. How one image is classified.", small),
        P(
            "The backbone's convolution layers pick up patterns such as texture, blur and reflections; some of "
            "these tend to differ between a real face and a photo of a print or a screen. The final score is the "
            "model's estimate of how likely the image is a spoof. The cut-off was chosen on the validation "
            "images, not the test images."
        ),
        P(
            "<b>Two sizes.</b> <i>Lite</i> is MobileNetV3-Small (1.5 million learned numbers, called "
            "parameters, and a 6 MB file). <i>Accurate</i> is MobileNetV3-Large (4.2 million parameters, "
            "17 MB). Fewer parameters mean a smaller, faster model."
        ),
        P(
            "<b>Other models I considered, and why not.</b> I did not train these; the reasons are about "
            "size, speed and effort:"
        ),
        B(
            "<b>ResNet-18 / ResNet-50:</b> reliable, but about 11 and 25 million parameters against 1.5 million "
            "for Lite. Too heavy for a task that asks for low cost, and Lite already scores well."
        ),
        B(
            "<b>Vision Transformers:</b> large, and they usually need much more training data than the "
            "12,000 images I have."
        ),
        B(
            "<b>MobileNetV2 and EfficientNet:</b> similar small models. I chose MobileNetV3 because it is a "
            "newer design made for fast inference on ordinary processors, and a pretrained version comes "
            "ready-made in the standard library I used (PyTorch/torchvision)."
        ),
        B(
            "<b>Classic methods</b> (hand-made texture features with a simple classifier): very cheap, but "
            "usually weaker on photos from many different phones."
        ),
        B(
            "<b>Specialised anti-spoofing networks</b> that also learn face depth: can be strong, but need "
            "extra labels and a more complex setup that this task does not call for."
        ),
        B("<b>Training from scratch:</b> needs far more data and time than a pretrained base."),
        P(
            "<b>How it was trained.</b> About eight passes (epochs) over the training images. Live images are "
            "rarer, so mistakes on them counted more. Small random changes (crop, flip, colour) were applied "
            "to the training images so the model does not simply memorise them."
        ),
        P("4. How I measured it", h2),
        B(
            "<b>Accuracy:</b> the share of test images classified correctly. Alone it is misleading here: a "
            "model that always answered &quot;spoof&quot; would be right 89.7% of the time."
        ),
        B(
            "<b>ROC-AUC:</b> picks one random live and one random spoof image and asks how often the model "
            "gives the spoof the higher spoof score. 0.5 is guessing, 1.0 is perfect. It does not depend on "
            "the cut-off."
        ),
        B(
            "<b>Two plain error rates:</b> live images wrongly flagged as spoof, and spoofs wrongly accepted "
            "as live. They show the two kinds of mistake separately."
        ),
        P("5. Results", h2),
        P(
            "All numbers are on the test images (389 live, 3,377 spoof). Speed is the time for one image on one "
            "laptop CPU core, so treat it as approximate."
        ),
        results,
        Spacer(1, 6),
        P(
            "<b>What I tried to improve it.</b> Training longer gave no gain: the model learns most of what it "
            "can within a few passes. Smaller images (160 px) were cheaper but slightly worse. Dropping some of "
            "the random image changes made no real difference, so the simpler set was used for Lite. The larger "
            "MobileNetV3 gave the best results. That is why there are two versions: Lite for low cost, "
            "Accurate for the best results. I picked Accurate because it scored best on the test images, so "
            "its test numbers may be slightly optimistic."
        ),
        P(
            "<b>Compression.</b> I tried shrinking the model further with INT8 quantisation (storing the "
            "numbers in a smaller format). Accuracy collapsed to chance level and it was not faster on my "
            "laptop, so I did not use it. The Lite model is already small and fast."
        ),
        P("6. How it is used", h2),
        P(
            "Both models are saved in ONNX, a standard model file format that runs in many environments. A "
            "command-line tool classifies image files, and the browser demo runs the same model inside the web "
            "page, so an uploaded image never leaves the user's device. The demo is hosted on Vercel and "
            "updates automatically when the code changes. The code has 28 automated tests, and GitHub checks "
            "it on every change."
        ),
        P("7. Limits to keep in mind", h2),
        B(
            "The model was tested only on this dataset. It may do worse with other cameras or kinds of attack."
        ),
        B(
            "The official training and test lists share 12 person IDs, so the test results may be slightly "
            "better than on completely new people. I kept the official test list for comparability."
        ),
        B(
            "The model expects a tightly cropped face like the training images. Whole photos give unreliable "
            "scores."
        ),
        B("Each setting was trained once, so small differences between runs are not meaningful."),
        P(
            "<b>Next steps:</b> test on a second dataset, add a face detector in front of the model, and "
            "repeat training with several random seeds."
        ),
        P("8. Running it", h2),
        P(
            "<font face='Courier' size='8.5'>pip install -r requirements.txt<br/>"
            "pip install -e .<br/>"
            "python -m liveness.predict face.png --model models/lite/model_fp32.onnx</font>"
        ),
        P("The README explains how to retrain from the dataset and how to run the tests."),
    ]

    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=1.9 * cm,
        rightMargin=1.9 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Lightweight genuine-vs-spoof image classification",
        author=AUTHOR,
    )
    doc.build(story)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="report.pdf")
    parser.add_argument("--demo-url", default=DEMO_URL)
    args = parser.parse_args()
    build(Path(args.out), args.demo_url)


if __name__ == "__main__":
    main()
