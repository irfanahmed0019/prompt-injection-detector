"""Load the deepset/prompt-injections dataset (Apache-2.0), cached as JSON in data/."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
URL = ("https://datasets-server.huggingface.co/rows?dataset=deepset/prompt-injections"
       "&config=default&split={split}&offset={off}&length=100")
SIZES = {"train": 546, "test": 116}


def _download(split):
    import urllib.request
    rows = []
    for off in range(0, SIZES[split], 100):
        with urllib.request.urlopen(URL.format(split=split, off=off), timeout=60) as r:
            rows += [x["row"] for x in json.load(r)["rows"]]
    return rows


def load(split):
    path = DATA / f"{split}.json"
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        path.write_text(json.dumps(_download(split), ensure_ascii=False), encoding="utf-8")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r["text"] for r in rows], [r["label"] for r in rows]
