import os
import subprocess
import huggingface_hub
import json
import threading
import tkinter as tk
from tkinter import scrolledtext, font
import cv2
from PIL import Image, ImageTk

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
# THE GUI APPLICATION (CAMERA KIOSK MODE)
# ==========================================
class CogniBridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CogniBridge AI Scanner")
        
        # Fullscreen setup for Raspberry Pi
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg="#000000") # Black background for camera UI
        
        # Setup paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.photos_dir = os.path.join(self.script_dir, "..", "data", "photos")
        os.makedirs(self.photos_dir, exist_ok=True) 
        
        # State variables
        self.frozen = False
        self.current_frame = None

        # Fonts
        self.btn_font = font.Font(family="Helvetica", size=20, weight="bold")
        self.text_font = font.Font(family="Helvetica", size=16)

        # Initialize Camera
        self.cap = cv2.VideoCapture(0)
        
        self.setup_ui()
        self.update_video_feed()

    def setup_ui(self):
        # Video Feed Label
        self.video_label = tk.Label(self.root, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Bottom Button Overlay Frame
        self.btn_frame = tk.Frame(self.root, bg="#000000")
        self.btn_frame.place(relx=0.5, rely=0.9, anchor=tk.CENTER)

        # Scan Button
        self.btn_scan = tk.Button(self.btn_frame, text="📸 SCAN DOCUMENT", font=self.btn_font, 
                                  bg="#a6e3a1", fg="black", height=2, width=20, 
                                  command=self.capture_and_process)
        self.btn_scan.pack(side=tk.LEFT, padx=20)

        # Exit Button
        self.btn_exit = tk.Button(self.btn_frame, text="❌ EXIT", font=self.btn_font, 
                                  bg="#f38ba8", fg="black", height=2, width=10, 
                                  command=self.on_exit)
        self.btn_exit.pack(side=tk.LEFT, padx=20)

    def update_video_feed(self):
        """Continuously pulls frames from the camera and updates the UI."""
        if not self.frozen:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv_img)
                self.photo = ImageTk.PhotoImage(image=pil_img)
                self.video_label.config(image=self.photo)

        self.root.after(15, self.update_video_feed)

    def capture_and_process(self):
        """Freezes the screen, saves the image, and starts the AI pipeline thread."""
        if self.current_frame is not None and not self.frozen:
            self.frozen = True 
            
            filepath = os.path.join(self.photos_dir, "scan.png")
            cv2.imwrite(filepath, self.current_frame)
            
            # Phase 1: OCR
            self.btn_scan.config(text="👁️ READING TEXT...", bg="#f9e2af", state=tk.DISABLED)
            
            # Fire off the combined AI pipeline in a background thread
            threading.Thread(target=self.run_full_pipeline, args=(filepath,), daemon=True).start()

    def run_full_pipeline(self, filepath):
        """Runs MindOCR, then feeds the result to Qwen (Running in background thread)."""
        # Step 1: Extract Text
        raw_text = run_mindocr_isolated(filepath)
        
        if not raw_text:
            # If OCR fails, jump straight to displaying the error
            self.root.after(0, self.display_results, "", "No text found in the image or OCR failed.")
            return

        # Step 2: Update UI to show we are switching from Vision to Language
        self.root.after(0, self.update_button_state, "🧠 SIMPLIFYING...", "#cba6f7")
        
        # Step 3: Simplify Text using Qwen
        simplified = cognibridge_simplify(raw_text)
        
        # Step 4: Send both results back to the main UI thread to be displayed
        self.root.after(0, self.display_results, raw_text, simplified)

    def update_button_state(self, text, color):
        """Helper to safely update the button from a background thread."""
        self.btn_scan.config(text=text, bg=color)

    def display_results(self, raw_text, simplified_text):
        """Builds an overlay showing the original text vs the simplified translation."""
        self.btn_scan.config(text="✅ DONE", bg="#89b4fa")

        # Create a dark overlay frame in the center of the screen
        self.result_frame = tk.Frame(self.root, bg="#1e1e2e", bd=5, relief=tk.RAISED)
        self.result_frame.place(relx=0.5, rely=0.45, anchor=tk.CENTER, relwidth=0.85, relheight=0.75)

        # --- Top Section: Original Text ---
        tk.Label(self.result_frame, text="📄 Original Legal Text:", font=self.btn_font, bg="#1e1e2e", fg="#f38ba8").pack(pady=(15, 5))
        
        raw_box = scrolledtext.ScrolledText(self.result_frame, font=self.text_font, wrap=tk.WORD, bg="#313244", fg="#cdd6f4", bd=0, height=5)
        raw_box.pack(fill=tk.X, padx=20, pady=5)
        raw_box.insert(tk.END, raw_text if raw_text else "N/A")
        raw_box.config(state=tk.DISABLED) # Read-only

        # --- Bottom Section: Simplified Translation ---
        tk.Label(self.result_frame, text="✨ CogniBridge Translation:", font=self.btn_font, bg="#1e1e2e", fg="#a6e3a1").pack(pady=(15, 5))
        
        simple_box = scrolledtext.ScrolledText(self.result_frame, font=self.btn_font, wrap=tk.WORD, bg="#313244", fg="#a6e3a1", bd=0, height=6)
        simple_box.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        simple_box.insert(tk.END, simplified_text)
        simple_box.config(state=tk.DISABLED) # Read-only

        # Reset Button to take another photo
        btn_reset = tk.Button(self.result_frame, text="🔄 Take Another Photo", font=self.btn_font, bg="#89b4fa", fg="black", command=self.reset_scanner)
        btn_reset.pack(pady=15)

    def reset_scanner(self):
        """Destroys the result overlay and unfreezes the camera feed."""
        self.result_frame.destroy()
        self.btn_scan.config(text="📸 SCAN DOCUMENT", bg="#a6e3a1", state=tk.NORMAL)
        self.frozen = False

    def on_exit(self):
        """Safely release the camera before closing."""
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CogniBridgeApp(root)
    root.mainloop()