# PiERN

**PiERN** is the official repository of the DREAMLAB-PKU team.  
This project provides code, data, and model implementations for our research work.

---

## 📌 Introduction

Tasks on complex systems often require **high-precision numerical computation** to support decision-making.  
However, current large language models (LLMs) struggle to natively integrate such computations as an intrinsic and interpretable capability. Multi-agent approaches can leverage external experts, but they suffer from **communication overhead** and **limited scalability**.

To address this, we propose **Physically-isolated Experts Routing Network (PiERN)**, an architecture for integrating **computation and reasoning**. Unlike tool-use workflows or function-calling, PiERN **endogenously integrates computational modules into neural networks**. After separately training experts, a text-to-computation module, and a router, PiERN performs reasoning and computation at the **token level**, enabling iterative alternation within a single chain of thought.

We evaluate PiERN on both **linear and nonlinear numerical reasoning tasks**, against LLM finetuning and multi-agent systems. Results show that PiERN achieves not only higher accuracy but also **significant improvements** in:
- Response latency  
- Token usage  
- GPU energy consumption  

PiERN offers an **efficient, interpretable, and scalable** paradigm for interfacing language models with scientific systems.

📄 For more details, please see and cite our [PiERN Paper](./PiERN.pdf).  
👉 Project page: https://github.com/DREAMLAB-PKU/PiERN


## 🚀 Quick Start

Clone the repository and install dependencies, then enter the code directory and make sure all the required models are downloaded (see the corresponding *.py files for download instructions):

```bash
git clone https://github.com/DREAMLAB-PKU/PiERN.git
cd PiERN
pip install -r requirements.txt
cd code
python3 capacity_sample_PiERN.py

