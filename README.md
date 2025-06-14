# UnDi: Uncertainty and Diversity Based Selection for Active Learning in Vision-Language Models

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official PyTorch implementation for **UnDi**, a novel active learning method for vision-language models that strategically selects samples based on both uncertainty and diversity.

## 🚀 Abstrate

Pre-trained vision-language models (VLMs) such as CLIP demonstrate strong zero-shot capabilities, yet they still underperform fully supervised models in domain-specific vision tasks. Model adaptation is an effective approach that reduces this performance gap with lower computational cost by fine-tuning only a small subset of model parameters while keeping the pre-trained backbone frozen, while selecting appropriate samples to further enhance VLM performance remains an ignored barrier. Active Learning (AL) offers a promising solution by selecting a small subset of unlabeled data for annotation. Previous methods mainly focus on uncertainty or diversity. However, they fail to balance both at the same time. To solve this, in this work, we propose UnDi, a novel method that achieves both **Un**certainty and **Di**versity for selecting samples. First, UnDi introduces a scoring mechanism that jointly considers sample confidence, variability, and entropy to assess sample uncertainty. Second, to further promote diversity and avoid redundancy, UnDi applies K-Means clustering on high-uncertainty candidate samples to gain sample diversity. Furthermore, we regard the AL question as a semi-supervised learning problem as there are many unlabeled samples available. To better utilize the knowledge inside unlabeled sample, UnDi leverages CLIP to generate pseudo-labels together with confidence-based filter to ensure the quality of pseudo labels. Extensive experiments on multiple image classification datasets show that UnDi significantly outperforms existing AL baselines under both with and without unlabeled samples scenarios.


**Framework**
<p align="center">
  <img src="framework.jpg" alt="UnDi Framework" width="800">
</p>
Figure: Overview of the UnDi framework showing the uncertainty and diversity-based selection process for active learning in vision-language models.

## ✨ Key Features

- **Multi-dimensional Uncertainty Scoring**: Integrates sample entropy, prediction confidence, and variability for robust uncertainty evaluation
- **Two-stage Selection Strategy**: First identifies high-uncertainty candidates, then applies K-means clustering to ensure diversity
- **Pseudo-labeling Framework**: Leverages CLIP's zero-shot capabilities to transform active learning into a semi-supervised setting
- **Superior Performance**: Outperforms existing AL baselines across multiple image classification datasets

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0.1+
- CUDA (recommended for GPU acceleration)

We recommend running the code in a Ubuntu environment or Windows Subsystem for Linux (WSL).

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Yangfan-123-cell/UnDi.git
cd UnDi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Datasets

Please refer to [DATASETS.md](DATASETS.md) for detailed instructions on downloading and preparing the datasets.

**Supported Datasets:**
- EuroSAT
- Oxford Pets
- DTD (Describable Textures Dataset)
- Caltech101
- Flowers102
- StanfordCars
- FGVC-Aircraft

Ensure your datasets are organized within the `./DATA` directory as specified in the configuration files.

## 🚀 Quick Start

### Running UnDi

Run experiments using the main script located in `/scripts/alvlm/`:

```bash
bash ./scripts/alvlm/main.sh <DATASET> <CFG> <ALMETHOD> <SEED>
```

**Parameters:**
- `DATASET`: Dataset name (`eurosat`, `oxford_pets`,`oxford_flowers`, `fgvc_aircraft`,`dtd`,`caltech101`,`sandford_cars`)
- `CFG`: Configuration file (`vit_b32,vit_b16,rn50,rn101`)
- `ALMETHOD`: Active learning method (`learnability` for our method)
- `SEED`: Random seed for reproducibility


### Examples

**Run UnDi on EuroSAT dataset:**
```bash
bash ./scripts/alvlm/main.sh eurosat vit_b32 learnability 1
```

**Run baseline methods:**
```bash
# Entropy-based selection
bash ./scripts/alvlm/main.sh eurosat vit_b32 entropy 1

# Random selection
bash ./scripts/alvlm/main.sh eurosat vit_b32 random 1

# CoreSet method
bash ./scripts/alvlm/main.sh eurosat vit_b32 coreset 1

# BADGE method
bash ./scripts/alvlm/main.sh eurosat vit_b32 badge 1
```

## 📁 Important Projects Structure

```
UnDi/
├── configs/                 # Configuration files
│   ├── datasets/           # Dataset configurations
│   └── trainers/           # Model and training configurations
    ...
├── scripts/                # Execution scripts
│   └── alvlm/              # Main.sh
│   └── coop/               # Prompt tuning
    ...
├── trainers/               
│   └── _ptcache_/          
│   └── active_learning/    # Active learning approaches(you can add your project in it)
│   └── alvlm.py/           # Main code for this project(you can change the way of training in it)
    ...
├── DATA/                   # You can put your datasets in it
├── requirements.txt        # Python dependencies
├── DATASETS.md             # The links of all datasets
|   ...
└── README.md               # This file
```

## ⚙️ Configuration

The main script configures the following key parameters:

- `--trainer ALVLM`: Active Learning Vision-Language Model trainer
- `TRAINER.COOP.N_CTX 16`: Number of context tokens
- `TRAINER.COOP.CSC True`: Class-specific context
- `TRAINER.COOP.CLASS_TOKEN_POSITION "end"`: Class token position
- `TRAINER.COOPAL.METHOD`: Active learning strategy
- `TRAINER.COOPAL.GAMMA 0.1`: Pseudo-labeling confidence threshold

## 📈 Results

Results including logs and model checkpoints are saved in:
```
output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}_al${ALMETHOD}_mode${MODE}/seed${SEED}
```



⭐ If you find this project useful, please consider giving it a star!

This work builds upon the [PCB](https://github.com/kaist-dmlab/pcb) project - we appreciate their contributions to the field
