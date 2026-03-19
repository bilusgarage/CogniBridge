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

*Note for Evaluators: This project was developed **entirely** on macOS (Apple Silicon). If you are testing this on Windows or Linux, please refer to the [Official MindSpore Installation Guide](https://www.mindspore.cn/install/en) to ensure you install the correct hardware-specific version of MindSpore for your machine before running the setup commands below. MindSpore version 2.5.0 can be found [here](https://www.mindspore.cn/versions/en/).*

---

## 🚀 Installation & Setup

### TBA

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.