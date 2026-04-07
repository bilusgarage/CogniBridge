# 🌉 CogniBridge

CogniBridge is an AI-powered text simplification engine designed to break down language barriers. Built for Windows, Linux, and macOS, this application translates complex, jargon-heavy English—such as legal contracts, corporate communications, and academic papers—into plain, easily accessible language. Whether you are a professional needing to parse dense documents quickly or a user looking for clear explanations, CogniBridge bridges the gap between complex text and pure comprehension.

It leverages **MindSpore**, **MindNLP** (powering a local Qwen 2 Instruction model), and **MindOCR** to process both digital text and scanned physical documents.

Looking for the mobile version? Check out the [cognibridge_mobile repository](https://github.com/bilusgarage/cognibridge_mobile)

## 🧠 Architecture

We use two separate Conda environments:
1. **`cogni39` (The NLP Brain):** Runs MindSpore 2.7.0 and handles the text-to-text generative AI using Qwen2-0.5B-Instruct.
2. **`mindocr_env` (The Vision Brain):** Runs MindSpore 2.5.0 and handles optical character recognition for scanned PDFs, JPEGs and PNGs.

The main script seamlessly bridges these environments by delegating OCR tasks to the isolated vision environment when necessary.

---

## ⚙️ Prerequisites
Before installing, ensure your system has the following:
* **Conda** (Miniconda suggested)\
Install on Debian through:\
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh`\
`bash Miniconda*.sh`\
For other systems: https://www.anaconda.com/docs/getting-started/miniconda/install
* **Git**
* **[MindSpore 2.5.0 wheel installation package](https://www.mindspore.cn/versions/en/)**

Tested on:
* macOS 26 (ARM64)
* Raspberry Pi OS (Debian 13 Trixie/ARM64)

---

## 🚀 Installation Guide

**1. Clone the CogniBridge repository to your desired location**

`git clone https://github.com/bilusgarage/cognibridge`

**2. Download [MindSpore 2.5.0 (Python 3.9) package](https://www.mindspore.cn/versions/en/#2.5.0) for your system**

Example:
* for Windows x86-64: `mindspore-2.5.0-cp39-cp39-win_amd64.whl`
* for Linux ARM: `mindspore-2.5.0-cp39-cp39-linux_aarch64.whl` 

**2. Move the `mindspore[...].whl` file to the path `CogniBridge/mindspore_installation_package/`**

**3. Launch installation script `install.py`**

## 🤔 How to use it?

	1. Launch `src/CogniBridge.py`
	2. Point your camera at the complex document or text
	3. Press "Scan document" to take a picture
	4. Wait
	5. You will see the simplified text overlayed on top of the photo. CogniBridge will read the text out loud, while highlighting each currently read sentence. You can control the TTS using playback buttons.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Project for Huawei ICT Competition 2026, Innovation Track.
Made by Bartłomiej Bil, Magdalena Błaszkowska, Weronika Strzyżewska