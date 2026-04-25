# Function-word De-Attention
# Pay Less Attention to Function Words for Free Robustness of Vision-Language Models

This repository contains the official PyTorch implementation for the paper [[ArXiv]](https://arxiv.org/abs/2512.07222):

> **Pay Less Attention to Function Words for Free Robustness of Vision-Language Models**  
> *ICLR 2026*  
> Qiwei Tian, Chenhao Lin, Zhengyu Zhao, Chao Shen

We propose **Function-word De-Attention (FDA)**, a lightweight and training-free (or low-cost fine-tuning) mechanism that improves the adversarial robustness of Vision-Language Models (VLMs) by reducing spurious attention on function words (e.g., *is, are, of, the*).

---

## 🔍 Overview

Modern VLMs often exhibit a trade-off between **robustness** and **clean performance**, especially under cross-modal adversarial attacks.  
Our key observation is:

> **Function words act as robustness vulnerabilities in vision–language alignment.**

Based on this, we introduce **FDA**, which:
- Computes function-word-specific cross-attention,
- Treats it as a disturbance term,
- And differentially subtracts it from the original attention.

FDA can be plugged into existing VLMs **without adversarial training**, yielding significant robustness gains with negligible performance drop.

## ⚙️ Environment Setup

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

We use Flickr30k/COCO/RefCOCO+ for T2IR and VG evaluation. These datasets can be accessed only.

## Finetune using FDA

To fine-tune ALBEF model on Retrieval:

```
bash ir_train.sh
```

*Note: the script uses Flickr30k as default, to train on MS-COCO, change the value of $data from `flickr` to `coco`*.

To fine-tune ALBEF model on Visual Grounding:

```
bash vg_train.sh
```

### 24/04/2026  Update 
🎉 **Our ur code (fine-tune) has been released! Code for evaluation will also be updated early May.**

### 28/01/2026  Update 
🏆 **Our paper has been accepted by ICLR26!** Code will be updated shortly.**

##To-do
1. Add evaluation code for robustness evaluation.
2. ...



