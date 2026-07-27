from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from predict import predict, create_models
from data.data_setup import class_names

from PIL import Image
import matplotlib.pyplot as plt


image_dir = Path(__file__).parent / "images"


for image_path in image_dir.iterdir():
    if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    results = predict(image_path)

    title = ""

    for model_name, (prediction, confidence) in results.items():
        title += f"{model_name}: {prediction} (confidence:{confidence:.2f}%)\n"

    image = Image.open(image_path)

    plt.figure(figsize=(6,6))

    plt.imshow(image)

    plt.title(
        title,
        fontsize=11,
        pad=20
    )

    plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    results_dir = ROOT / "results" / "predictions"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        results_dir / f"{image_path.stem}_prediction.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"{image_path.name:.20}"
        f"{prediction:10}"
        f"{confidence:.2f}%"
    )