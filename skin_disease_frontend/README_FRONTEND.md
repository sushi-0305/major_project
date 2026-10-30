# Skin Disease Classification Frontend

Simple Flask frontend for the HAM10000 few-shot skin disease classification project.

## Structure

```text
app.py
templates/
    index.html
static/
    style.css
    script.js
```

## Run

From the project root:

```bash
source venv/bin/activate
pip install flask
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The app expects the trained checkpoint at:

```text
checkpoints/ham10000_prototypical/proto_networks_best.pth
```

It uses five fixed support images per class from `data/splits/train.csv`.

> Research/demo interface only. Predictions should not be used as medical advice.
