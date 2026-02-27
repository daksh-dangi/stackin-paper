# Stacking Paper: Building on ML Research

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active_Research-success.svg)]()

Welcome to **Stacking Paper(s)**! This repository is my personal playground for breaking down the latest machine learning research papers that catch my eye, implementing them from scratch, and bridging the gap between these intertwining topics to further improve them - all while documenting my findings along the way in the hopes that somebody might find it useful :)

Rather than just posting finished code, I use this space to write articles detailing my implementation struggles, architectural decisions, and the "gotchas" that aren't mentioned in the original papers.

## 🗂️ Table of Contents & Research Logs

Below is a running index of the papers I have implemented and the corresponding articles/notes I've written about them.

| Date | Paper / Topic | Key Concepts Explored | Implementation & Article |
| :--- | :--- | :--- | :--- |
| **Feb 2026** | *Verifiable Reward RL on Micro-Datasets* | GRPO, LLM-as-a-Judge, Math Reasoning | [📁 View Project & Article](./2026_GRPO_Micro_Datasets) |
| **Upcoming** | *DeepSeek-R1 Distillation* | RLHF, KD (Knowledge Distillation) | *Work in Progress* |
| **Upcoming** | *[Insert Next Paper Name]* | [Insert Concept] | *Planned* |

---

## 🏗️ Repository Structure

Because machine learning dependencies change rapidly between different research papers, **this is a monorepo where each project is entirely isolated.** 
```text
the-paper-trail/
│
├── README.md                      <- You are here!
├── shared_utils/                  <- Custom data loaders, API wrappers, and metric functions
│
├── 2026_GRPO_Micro_Datasets/      <- Example Project Folder
│   ├── README.md                  <- 📝 The article, findings, and project-specific notes
│   ├── requirements.txt           <- ⚙️ Strict, isolated dependencies for this project
│   ├── src/                       <- 💻 The actual implementation code
│   └── data/                      <- 📊 Datasets (or scripts to fetch them)
│
└── ...