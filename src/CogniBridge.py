import os
import subprocess
import huggingface_hub
import json
import threading
import tkinter as tk
from tkinter import scrolledtext, font

# Monkey patch
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import mindspore as ms
import mindnlp
from transformers import pipeline
from datasets import load_dataset

print("Initializing CogniBridge AI... (Loading dataset and model)")

# 1. Load the dataset
dataset = load_dataset("waboucay/wikilarge", 'original', split="train")

ex1_complex = dataset[0]['complex']
ex1_simple = dataset[0]['simple']
ex2_complex = dataset[1]['complex']
ex2_simple = dataset[1]['simple']
ex3_complex = dataset[2]['complex']
ex3_simple = dataset[2]['simple']

# 2. Custom PARAGRAPH
ex4_paragraph_complex = "In the event that the Purchaser fails to remit payment in full within the stipulated timeframe of thirty (30) days from the date of invoice issuance, the Vendor reserves the explicit right to suspend all ongoing services and impose a late penalty fee of one and one-half percent (1.5%) per month on the outstanding balance. Furthermore, any subsequent legal costs incurred during the collection process shall be borne entirely by the Purchaser."
ex4_paragraph_simple = "If the buyer doesn't pay the full money in 30 days after getting the invoice, the seller can pause all services and charge a 1.5% monthly fee on the unpaid money. The buyer must pay for any legal fees needed to collect the money."

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2-0.5B-Instruct",
    dtype=ms.float32
)

print("CogniBridge is ready!\n")

def cognibridge_simplify(text):
    word_count = len(text.split())
    dynamic_max_tokens = min(int((word_count * 2) + 20), 512)
    prompt = f"""You are an expert at simplifying complex English. Look at these examples of complex text being rewritten into simple text.

Complex: {ex1_complex}
Simple: {ex1_simple}

Complex: {ex2_complex}
Simple: {ex2_simple}

Complex: {ex3_complex}
Simple: {ex3_simple}

Complex: {ex4_paragraph_complex}
Simple: {ex4_paragraph_simple}

Now, shorten and simplify this text using the simple sentence style. ONLY output the simplified text. Do not add any explanations, notes, or extra text.
Complex: {text}
Simple:"""

    result = pipe(
        prompt,
        max_new_tokens=dynamic_max_tokens,
        return_full_text=False,
        temperature=0.1
    )

    raw_output = result[0]["generated_text"].strip()
    return raw_output.split("Complex:")[0].strip()

def run_mindocr_isolated(image_path):
    mindocr_python_path = "/opt/miniconda3/envs/mindocr_env/bin/python"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    predict_script = os.path.join(project_root, "mindocr", "tools", "infer", "text", "predict_system.py")
    
    command = [
        mindocr_python_path, predict_script,
        "--image_dir", image_path,
        "--det_algorithm", "DB++",
        "--rec_algorithm", "CRNN"
    ]
    
    subprocess.run(command, cwd=project_root)
    results_file = os.path.join(project_root, "inference_results", "system_results.txt")
    extracted_text = ""
    
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split('\t')
                if len(parts) > 1:
                    raw_json_string = parts[1].strip()
                    try:
                        data = json.loads(raw_json_string)
                        processed_boxes = []
                        for item in data:
                            points = item['points']
                            min_x = min(p[0] for p in points)
                            min_y = min(p[1] for p in points)
                            max_y = max(p[1] for p in points)
                            processed_boxes.append({
                                'text': item['transcription'],
                                'min_x': min_x,
                                'center_y': (min_y + max_y) / 2.0,
                                'height': max_y - min_y
                            })
                        processed_boxes.sort(key=lambda b: b['center_y'])
                        lines = []
                        current_line = []
                        for box in processed_boxes:
                            if not current_line:
                                current_line.append(box)
                            else:
                                prev_box = current_line[-1]
                                if abs(box['center_y'] - prev_box['center_y']) < max(box['height'], prev_box['height']) * 0.5:
                                    current_line.append(box)
                                else:
                                    lines.append(current_line)
                                    current_line = [box]
                        if current_line:
                            lines.append(current_line)
                        ordered_words = []
                        for line_group in lines:
                            line_group.sort(key=lambda b: b['min_x'])
                            for box in line_group:
                                ordered_words.append(box['text'])
                        extracted_text += " ".join(ordered_words) + " "
                    except json.JSONDecodeError:
                        extracted_text += raw_json_string + " "
    return extracted_text.strip()


# ==========================================
# THE GUI APPLICATION
# ==========================================
class CogniBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CogniBridge AI")
        
        # Make it fullscreen for the Raspberry Pi display
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#1e1e2e") # Modern dark background
        
        # Setup paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_txt = os.path.join(self.script_dir, "..", "data", "complex_text.txt")
        self.input_img = os.path.join(self.script_dir, "..", "data", "scan.png")

        # Fonts
        self.title_font = font.Font(family="Helvetica", size=24, weight="bold")
        self.text_font = font.Font(family="Helvetica", size=14)
        self.btn_font = font.Font(family="Helvetica", size=16, weight="bold")

        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="🧠 CogniBridge Edge AI", font=self.title_font, bg="#1e1e2e", fg="#cdd6f4", pady=20)
        header.pack(fill=tk.X)

        # Main Content Frame (Split into left/right)
        content_frame = tk.Frame(self.root, bg="#1e1e2e")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left Column: Original Text
        left_frame = tk.Frame(content_frame, bg="#1e1e2e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        tk.Label(left_frame, text="Original Input", font=self.btn_font, bg="#1e1e2e", fg="#f38ba8").pack(anchor="w")
        self.original_box = scrolledtext.ScrolledText(left_frame, font=self.text_font, wrap=tk.WORD, bg="#313244", fg="#cdd6f4", bd=0, padx=10, pady=10)
        self.original_box.pack(fill=tk.BOTH, expand=True)

        # Right Column: Simplified Output
        right_frame = tk.Frame(content_frame, bg="#1e1e2e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        tk.Label(right_frame, text="Simplified Output", font=self.btn_font, bg="#1e1e2e", fg="#a6e3a1").pack(anchor="w")
        self.simplified_box = scrolledtext.ScrolledText(right_frame, font=self.text_font, wrap=tk.WORD, bg="#313244", fg="#cdd6f4", bd=0, padx=10, pady=10)
        self.simplified_box.pack(fill=tk.BOTH, expand=True)

        # Bottom Buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e2e", pady=20)
        btn_frame.pack(fill=tk.X)

        self.btn_text = tk.Button(btn_frame, text="📄 Process Document", font=self.btn_font, bg="#89b4fa", fg="black", command=self.start_text_thread, height=2, width=20)
        self.btn_text.pack(side=tk.LEFT, expand=True, padx=10)

        self.btn_img = tk.Button(btn_frame, text="👁️ Process Image Scan", font=self.btn_font, bg="#cba6f7", fg="black", command=self.start_image_thread, height=2, width=20)
        self.btn_img.pack(side=tk.LEFT, expand=True, padx=10)

        self.btn_exit = tk.Button(btn_frame, text="❌ Exit Fullscreen", font=self.btn_font, bg="#f38ba8", fg="black", command=self.root.destroy, height=2, width=15)
        self.btn_exit.pack(side=tk.RIGHT, expand=True, padx=10)

    # --- Threading to keep UI responsive ---
    def start_text_thread(self):
        self.update_boxes("Reading text document...", "Waiting for AI translation...")
        threading.Thread(target=self.process_document_gui, daemon=True).start()

    def start_image_thread(self):
        self.update_boxes("Waking up MindOCR vision engine...", "Waiting for AI translation...")
        threading.Thread(target=self.process_image_gui, daemon=True).start()

    def update_boxes(self, left_text, right_text):
        """Helper to update text boxes safely."""
        self.original_box.delete(1.0, tk.END)
        self.original_box.insert(tk.END, left_text)
        self.simplified_box.delete(1.0, tk.END)
        self.simplified_box.insert(tk.END, right_text)
        self.root.update()

    # --- GUI-Adapted Logic ---
    def process_document_gui(self):
        if not os.path.exists(self.input_txt):
            self.update_boxes("Error", f"Could not find {self.input_txt}")
            return
            
        with open(self.input_txt, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
            
        self.update_boxes(raw_text, "Processing with Qwen... Please wait.")
        simplified = cognibridge_simplify(raw_text)
        self.update_boxes(raw_text, simplified)

    def process_image_gui(self):
        if not os.path.exists(self.input_img):
            self.update_boxes("Error", f"Could not find {self.input_img}")
            return

        # Phase 1: OCR
        self.update_boxes("Scanning Image with DB++ & CRNN...\n\nExtracting spatial coordinates...", "Waiting for vision engine...")
        raw_text = run_mindocr_isolated(self.input_img)
        
        if not raw_text:
            self.update_boxes("Error", "No text found in image or OCR failed.")
            return

        # Phase 2: NLP
        self.update_boxes(f"[Extracted from Image]:\n\n{raw_text}", "Translating jargon to simple English...\n\nPlease wait.")
        simplified = cognibridge_simplify(raw_text)
        
        # Final Output
        self.update_boxes(f"[Extracted from Image]:\n\n{raw_text}", simplified)

if __name__ == "__main__":
    # The models load FIRST in the terminal, then the UI launches.
    root = tk.Tk()
    app = CogniBridgeApp(root)
    root.mainloop()