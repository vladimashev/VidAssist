# VidAssist: LLM-based Goal-Oriented Planning in Instructional Videos

Non-official implementation of the paper:  
**"Propose, Assess, Search: Harnessing LLMs for Goal-Oriented Planning in Instructional Videos"**  
[arXiv:2409.20557](https://arxiv.org/abs/2409.20557)

---

## 📌 Overview

**VidAssist** is a modular framework designed to perform goal-oriented reasoning in instructional videos using Large Language Models (LLMs).  
It supports zero-shot and few-shot planning and operates in three stages:

1. **Propose** – Generate candidate actions using an LLM.
2. **Assess** – Evaluate candidate actions using scoring functions.
3. **Search** – Use breadth-first search to select optimal action sequences.

The framework is evaluated on standard datasets like **COIN** and **CrossTask**.

---

## 🧠 Key Features

- 🤖 LLM-based semantic planning.
- 🔍 Breadth-first search over plan proposals.
- 🧪 Zero/few-shot performance without full supervision.
- 🧩 Modular design: you can plug in your own LLM or scoring functions.

---

## 📂 Project Structure

