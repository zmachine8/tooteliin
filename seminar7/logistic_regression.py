from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


# Tooteklasside nimed samas jarjekorras, nagu tahame neid valjundis naha.
PRODUCTS = ["empty", "kalkun", "rulaad", "salami", "veis"]
MODEL_NAME = "facebook/dinov2-small"
BASE_DIR = Path(__file__).resolve().parent
template_count = 20
batch_size = 16

# Kui GPU on olemas, kasutame seda. Muidu tootame CPU-l.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Need objektid teisendavad pildi DINOv2 vektoresituseks.
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def create_embeddings(paths):
    embeddings = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        batch_embeddings = F.normalize(batch_embeddings, dim=-1).cpu().numpy()
        embeddings.extend(batch_embeddings)

    return embeddings


def load_paths_and_split():
    # Tootekaustad asuvad selle faili korval.
    train_paths = []
    train_labels = []
    test_paths = {product: [] for product in PRODUCTS}

    for product in PRODUCTS:
        folder = BASE_DIR / product / "product_area"
        images = sorted(folder.glob("*.png"))

        # Esimesed template_count pilti lahevad treeninguks, ulejaanud testiks.
        for path in images[:template_count]:
            train_paths.append(path)
            train_labels.append(product)

        test_paths[product] = images[template_count:]

    return train_paths, train_labels, test_paths


# Koigepealt kogume failiteed kokku.
train_paths, train_labels, test_paths = load_paths_and_split()

# Teisendame treeningpildid DINOv2 vektoresitusteks.
train_features = np.array(create_embeddings(train_paths))

# Opetame vektoresituste peal logistilise regressiooni mudeli.
classifier = LogisticRegression(max_iter=2000,C=5.0)
classifier.fit(train_features, train_labels)

# Hindame mudelit testpiltidel ja prindime tulemused.
correct_percentages = []

for product in PRODUCTS:
    counts = {name: 0 for name in PRODUCTS}
    misclassifications = []
    total = len(test_paths[product])
    test_features = create_embeddings(test_paths[product])

    for path, feature in zip(test_paths[product], test_features):
        prediction = classifier.predict(np.array([feature]))[0]
        counts[prediction] += 1

        if prediction != product:
            misclassifications.append(f"{path.name} -> {prediction}")

    print(product)

    for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        percent = 100 * count / total if total > 0 else 0
        print(f"{name} ({percent:.1f}%, {count} images)")

    correct_percent = 100 * counts[product] / total if total > 0 else 0
    correct_percentages.append(correct_percent)

    if misclassifications:
        print("Valed ennustused:")
        for line in misclassifications:
            print(line)
    else:
        print("Valesid ennustusi ei olnud")

    print()

# Salvestame tulpdiagrammi, mis naitab iga klassi oigete ennustuste protsenti.
plot_path = BASE_DIR / f"classification_accuracy_logistic_regression_templates{template_count}.png"
plt.figure(figsize=(8, 5))
bars = plt.bar(PRODUCTS, correct_percentages, color="seagreen")
plt.ylim(0, 100)
plt.xlabel("Toote liik")
plt.ylabel("Oiged ennustused (%)")
plt.title("Tootetuvastuse tapsus logistilise regressiooniga")

for bar, percent in zip(bars, correct_percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        percent + 1,
        f"{percent:.1f}%",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.close()
print(f"Salvestasin graafiku faili: {plot_path}")
