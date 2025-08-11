

<!-- BEGIN_DIALECT_LINK -->

## Dialectal Audio Examples

**Listen in your browser:** 👉 [Dialect Examples (with audio players)](https://labspire.github.io/respin_did_interspeech25/examples.html)

<!-- END_DIALECT_LINK -->

# Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion

This repository contains the **official implementation** of the paper:

> **Saurabh Kumar, Amartyaveer, Prasanta Kumar Ghosh**  
> *Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion*  
> Interspeech 2025

---

## 📌 Overview

Dialectal variation poses a major challenge to **Automatic Speech Recognition (ASR)** systems, especially in linguistically diverse regions like India.  
Our work addresses this challenge by **jointly training ASR and Dialect Identification (DID)** models using **multimodal feature fusion**.

We propose **ASR-BN-ROB**, a novel architecture that combines:
- **Bottleneck Encoder** – captures acoustic and temporal dialect cues from speech.
- **RoBERTa Encoder** – processes ASR CTC embeddings to extract lexical and semantic cues.
- **Gating mechanism** – adaptively fuses speech and text features.
- **Feedback to ASR encoder** – without prepending dialect tokens, improving both ASR and DID.

Our approach achieves **state-of-the-art results** on the RESPIN dataset covering **8 Indian languages and 33 dialects**.

---

## 🚀 Key Features

- **Improved DID Accuracy**: 81.63% (highest among all evaluated systems)
- **Better ASR Performance**: CER = 4.65%, WER = 17.73%
- **Breaks the ASR–DID Trade-off**: Improves both tasks simultaneously
- **Robust to DID Errors**: Significant gains even when DID predictions are incorrect
- **Public Code & Models**: ESPnet recipes for reproducibility

---

## 📊 Dataset

We evaluate on the **RESPIN** dataset ([IISc, 2024](https://respin.iisc.ac.in/)):

| Language        | Code | # Dialects |
|-----------------|------|------------|
| Bhojpuri        | bh   | 3          |
| Bengali         | bn   | 5          |
| Chhattisgarhi   | ch   | 4          |
| Kannada         | kn   | 5          |
| Magahi          | mg   | 4          |
| Marathi         | mr   | 4          |
| Maithili        | mt   | 4          |
| Telugu          | te   | 4          |

- **Total dialects**: 33  
- **Training data**: ~140–175 hrs read speech per language

---

## 🏗 Model Architecture

- **Speech Features** → SSL Conformer Encoder → Bottleneck Encoder  
- **Text Features** → RoBERTa Encoder on ASR CTC embeddings  
- **Feature Fusion** → Gating mechanism + Attention Encoder  
- **Joint Optimization** → Hybrid CTC+Attention ASR loss + DID cross-entropy loss

---

## 📈 Results Summary

**Dialect Identification Accuracy (%):**

| System                | Avg. Accuracy |
|-----------------------|---------------|
| ASR-DID-ROB (baseline)| 80.74         |
| **ASR-BN-ROB (Proposed)** | **81.63** |

**ASR Performance:**

| System      | CER (%) | WER (%) |
|-------------|---------|---------|
| Base-ASR    | 4.81    | 18.38   |
| ASR-DID-ROB | 4.76    | 18.16   |
| **ASR-BN-ROB** | **4.65** | **17.73** |

---

## 📂 Repository Structure

```
respin_did_interspeech25/
│
├── espnet/           # ESPnet recipes for ASR+DID
├── samples/          # Example audio samples
├── scripts/          # Training and evaluation scripts
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

### ▶️ Training & Evaluation

Example command for training:

```bash
cd espnet/egs2/respin/asr1
./run.sh --stage 1 --stop_stage 12
```

For DID evaluation only:

```bash
python evaluate_did.py --config conf/did_config.yaml
```

---

## 📜 Citation

If you use this code or models in your work, please cite:

```bibtex
@inproceedings{kumar2025respin_did,
    title     = {Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion},
    author    = {Kumar, Saurabh and Amartyaveer and Ghosh, Prasanta Kumar},
    booktitle = {Proc. Interspeech},
    year      = {2025}
}
```

---

## 📬 Contact
- Saurabh Kumar – saurabhk0317@gmail.com
- SPIRE Lab, IISc Bangalore – spirelab.ee@iisc.ac.in
- Project Page – [RESPIN](https://respin.iisc.ac.in/)
- Code – [GitHub Repository](https://github.com/labspire/respin_did_interspeech25.git)