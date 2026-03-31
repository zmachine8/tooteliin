import os

import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


# Kasutame eeltreenitud DINOv2 mudelit, et teisendada pildid vektoresitusteks.
MODEL_NAME = "facebook/dinov2-small"

# template_count maarab, mitu esimest pilti votame igast klassist naidisteks.
# k maarab, mitu lahimat naidist ennustamisel arvesse votame.
template_count = 4
k = 1
batch_size = 16

# Kui GPU on olemas, kasutame seda. Muidu tootame CPU-l.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def create_embeddings(paths: list[str]) -> list[torch.Tensor]:
    embeddings = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        batch_embeddings = F.normalize(batch_embeddings, dim=-1).cpu()
        embeddings.extend(batch_embeddings)

    return embeddings


base_dir = os.path.dirname(__file__)

# Need on tooteklassid, mida tahame tuvastada.
products = ["kalkun", "rulaad", "salami", "veis", "empty"]

# Siia salvestame naidispiltide embeddingud.
templates = []

# Siia salvestame iga klassi pildifailide nimed.
product_images = {}

# Siia salvestame iga klassi oige tuvastuse protsendi.
correct_percentages = []

for product in products:
    folder = os.path.join(base_dir, product, "product_area")
    images = sorted([img for img in os.listdir(folder) if img.endswith(".png")])
    product_images[product] = images

    template_paths = [os.path.join(folder, img) for img in images[:template_count]]
    template_embeddings = create_embeddings(template_paths)

    for img, emb in zip(images[:template_count], template_embeddings):
        templates.append((product, emb))

for product in products:
    folder = os.path.join(base_dir, product, "product_area")
    counts = {name: 0 for name in products}
    total = 0
    misclassifications = []

    test_images = product_images[product][template_count:]
    test_paths = [os.path.join(folder, img) for img in test_images]
    test_embeddings = create_embeddings(test_paths)

    for img, emb in zip(test_images, test_embeddings):
        similarities = []

        # Vordleme testpilti koigi naidistega.
        for template_product, template_emb in templates:
            sim = float(torch.dot(emb, template_emb).item())
            similarities.append((sim, template_product))

        # Valime k koige sarnasemat naidist.
        neighbors = sorted(similarities, reverse=True)[:k]
        votes = {name: 0 for name in products}
        best_similarity = {name: -1.0 for name in products}

        # Loeme haaled kokku.
        for sim, template_product in neighbors:
            votes[template_product] += 1
            if sim > best_similarity[template_product]:
                best_similarity[template_product] = sim

        # Kui haaltearv on vordne, eelistame suurema sarnasusega klassi.
        best_product = max(products, key=lambda name: (votes[name], best_similarity[name]))
        counts[best_product] += 1
        total += 1

        if best_product != product:
            misclassifications.append(f"{img} -> {best_product}")

    # Prindime klassi tulemused ja mooned valed ennustused.
    print(product)
    for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        percent = 100 * count / total if total > 0 else 0
        print(f"{name} ({percent:.1f}%, {count} images)")

    correct_percent = 100 * counts[product] / total if total > 0 else 0
    correct_percentages.append(correct_percent)

    if misclassifications:
        print("Valede ennustuste naited:")
        for example in misclassifications[:5]:
            print(example)
    else:
        print("Valesid ennustusi ei olnud")
    print()

# Joonistame tulpdiagrammi, mis naitab iga klassi oigete ennustuste protsenti.
plot_path = os.path.join(base_dir, f"classification_accuracy_templates{template_count}_knn{k}.png")
plt.figure(figsize=(8, 5))
bars = plt.bar(products, correct_percentages, color="steelblue")
plt.ylim(0, 100)
plt.xlabel("Toote liik")
plt.ylabel("Oiged ennustused (%)")
plt.title("Tootetuvastuse tapsus")

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
