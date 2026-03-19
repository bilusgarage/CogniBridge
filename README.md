# 🌉 CogniBridge

CogniBridge is an AI-powered text simplification engine designed to translate highly complex English (such as legal, corporate, or academic documents) into plain, accessible language.

It leverages **MindSpore**, **MindNLP** (powering a local Qwen 2 Instruction model), and **MindOCR** to process both digital text and scanned physical documents.

## 🧠 Architecture

We use two separate Conda environments:
1. **`cogni39` (The NLP Brain):** Runs MindSpore 2.7.0 and handles the text-to-text generative AI using Qwen2-0.5B-Instruct.
2. **`mindocr_env` (The Vision Brain):** Runs MindSpore 2.5.0 and handles optical character recognition for scanned PDFs, JPEGs and PNGs.

The main script seamlessly bridges these environments by delegating OCR tasks to the isolated vision environment when necessary.

---

## ⚙️ Prerequisites
Before installing, ensure your system has the following:
* **Conda** (Miniconda or Anaconda)
* **Python 3.9.11**
* **Git**
* **MindSpore 2.5.0 wheel installation package

*Note for Evaluators: This project was developed **entirely** on macOS (Apple Silicon). If you are testing this on Windows or Linux, please get the MindSpore installation `.whl` file for your machine, for Python 3.9 before running the setup commands below. MindSpore version 2.5.0 can be found [here](https://www.mindspore.cn/versions/en/).* 

---

## 🚀 Installation Guide

1. Clone the CogniBridge repository to your desired location

`git clone https://github.com/bilusgarage/CogniBridge`

2. Download MindSpore 2.5.0 package for your system and Python 3.9
Example:
* for Windows x86-64: `mindspore-2.5.0-cp39-cp39-win_amd64.whl`
* for Linux ARM: `mindspore-2.5.0-cp39-cp39-linux_aarch64.whl` 

2. Move the `mindspore[...].whl` file to the folder `*CogniBridge/mindspore_installation_package/`

3. Launch installation file `install.py`

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.