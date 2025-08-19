<!-- BEGIN_DIALECT_LINK -->

## Dialectal Audio Examples

**Listen in your browser:** 👉 [Dialect Examples (with audio players)](https://labspire.github.io/respin_did_interspeech25/examples.html)

<!-- END_DIALECT_LINK -->

# Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion

Official implementation of the paper:

> **Saurabh Kumar, Amartyaveer, Prasanta Kumar Ghosh**  
> *[Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion](https://www.isca-archive.org/interspeech_2025/kumar25_interspeech.html#)*  
> Interspeech 2025

---

## 📌 Overview

Dialectal variation poses a major challenge to **Automatic Speech Recognition (ASR)** in Indian languages.  
This work addresses it by **jointly training ASR and Dialect Identification (DID)** models using **multimodal feature fusion**.

We propose **ASR-BN-ROB**, a joint architecture that combines:
- **Bottleneck Encoder** – captures acoustic and temporal cues.  
- **RoBERTa Encoder** – extracts lexical/semantic cues from ASR CTC embeddings.  
- **Gating Mechanism** – adaptively fuses speech and text features.  
- **Feedback to ASR Encoder** – improves ASR without prepending dialect tokens.  

---

## 🚀 Key Contributions

- **Improved DID**: 81.63% accuracy across 33 dialects  
- **Better ASR**: CER = 4.65%, WER = 17.73%  
- **Breaks the ASR–DID trade-off**: improves both tasks simultaneously  
- **Robustness**: maintains ASR performance even with incorrect DID predictions  
- **Reproducibility**: ESPnet-based code + pretrained models  

---

## 📊 Dataset

We evaluate on the **[RESPIN Corpus](https://spiredatasets.ee.iisc.ac.in/respincorpus)** (SPIRE Lab, IISc).  

- **Languages**: Bhojpuri, Bengali, Chhattisgarhi, Kannada, Magahi, Marathi, Maithili, Telugu  
- **Dialects**: 33 total  
- **Training data**: ~140–175 hrs read speech per language  
- **Setup**: Small train set is used for all languages in this work  

---

## 🏗 Model Architecture

- **Speech features** → SSL Conformer Encoder → Bottleneck Encoder  
- **Text features** → RoBERTa Encoder on CTC embeddings  
- **Fusion** → Gating mechanism + Attention Encoder  
- **Loss** → Hybrid CTC+Attention ASR loss + DID cross-entropy loss  

---

## 📈 Results

**Dialect Identification (Accuracy %):**

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
├── espnet_mods/ # Modified + new ESPnet files (commit 6b5f5269a2a6d8902fe3697f88ca9c0ccde35353)
│ ├── nets/...
│ ├── bin/...
│ ├── task/...
│ └── train/...
│
├── samples/ # Example audio samples
├── text_files/ # Example transcripts
├── run_all_asr_did.sh # Multi-language training + evaluation script
├── requirements.txt # Python dependencies (pip freeze)
├── examples.html # Demo page with audio players
└── README.md # This file
```


---

## ⚙️ Installation

1. **Clone ESPnet (specific commit):**
    ```bash
    git clone https://github.com/espnet/espnet.git
    cd espnet
    git checkout 6b5f5269a2a6d8902fe3697f88ca9c0ccde35353
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Copy modified files:**
    ```bash
    cp -r path_to_repo/espnet_mods/* espnet/
    ```

### ▶️ Training & Evaluation

Example command for training:

```bash
cd espnet/egs2/respin_did_is25/asr1
./run_all_asr_did.sh --stage 1 --stop_stage 13
```

---

## 📜 Citation

If you use this code or models in your work, please cite:

```bibtex
@inproceedings{kumar25_interspeech,
  title     = {{Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion}},
  author    = {{Saurabh Kumar and  Amartyaveer and Prasanta Kumar Ghosh}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{2770--2774}},
  doi       = {{10.21437/Interspeech.2025-421}},
  issn      = {{2958-1796}},
}
```

---

## 📬 Contact
- Saurabh Kumar – saurabhk0317@gmail.com
- SPIRE Lab, IISc Bangalore – spirelab.ee@iisc.ac.in
- Project Page – [RESPIN](https://respin.iisc.ac.in/)
- Code – [GitHub Repository](https://github.com/labspire/respin_did_interspeech25.git)