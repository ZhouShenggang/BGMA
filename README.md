# Efficient Image-text Retrieval via Bi-Cross-Graph Learning and Multi-Grained Alignment

This repository contains the official implementation of the paper:

**"Efficient Image-text Retrieval via Bi-Cross-Graph Learning and Multi-Grained Alignment"**

---

## Requirements

- Python 3.8
- PyTorch 2.0.1
- CUDA 11.7
- Additional dependencies listed in `requirements.txt`

To install dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

Download the dataset (e.g., MS-COCO, Flickr30K) and organize it as follows:
   ```
   data/
   ├── f30k_precomp/
   ├── coco_precomp/
   └── ...

---

## Training

To train the model, use:
```bash
python train.py
```

## Evaluation

To evaluate the trained model, use:
```bash
python test.py
```

---

## Results

Our model achieves state-of-the-art performance on benchmark datasets:
- **MS-COCO 1K**:
  - Image retrieval Recall@1: **81.1%**
  - Image retrieval Recall@5: **96.9%**
  - Image retrieval Recall@10: **99.5%**
  - Text retrieval Recall@1: **65.5%**
  - Text retrieval Recall@5: **92%**
  - Text retrieval Recall@10: **97.3%**
- **Flickr 30K**:
  - Image retrieval Recall@1: **80%**
  - Image retrieval Recall@5: **95.7%**
  - Image retrieval Recall@10: **98.5%**
  - Text retrieval Recall@1: **61.6%**
  - Text retrieval Recall@5: **86.3%**
  - Text retrieval Recall@10: **92.5%**

---
