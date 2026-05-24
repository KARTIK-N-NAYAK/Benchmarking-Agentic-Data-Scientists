from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ITEMS_FILE = BASE_DIR / "datasets.txt"
OUT_DIR = BASE_DIR / "descriptions"
OUT_DIR.mkdir(exist_ok=True)

api = KaggleApi()
api.authenticate()

items = [i.strip() for i in ITEMS_FILE.read_text().splitlines() if i.strip()]

for item in items:
    print(f"extracting description for {item} ...")

    title = None
    description = None

    if "/" in item:
        # DATASET
        meta = api.dataset_metadata(item)
        title = meta.title
        description = meta.description
    else:
        # COMPETITION (via list search)
        comps = api.competitions_list(search=item)
        match = next((c for c in comps if c.ref == item), None)

        if not match:
            print(f" Competition not found: {item}")
            continue

        title = match.title
        description = match.description

    md_path = OUT_DIR / f"{item.replace('/', '__')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(description or "_No description available._")

    print(f"saved → {md_path}")
