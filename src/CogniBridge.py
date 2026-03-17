import os
import huggingface_hub
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import mindspore as ms
import mindnlp
from transformers import pipeline
from datasets import load_dataset

print("Initializing CogniBridge AI... (Loading dataset and model)")

# 1. Load the dataset for our short sentence examples
dataset = load_dataset("waboucay/wikilarge", 'original', split="train")

ex1_complex = dataset[0]['complex']
ex1_simple = dataset[0]['simple']
ex2_complex = dataset[1]['complex']
ex2_simple = dataset[1]['simple']

# 2. Our custom PARAGRAPH example to teach it how to handle long legal text
ex3_paragraph_complex = "In the event that the Purchaser fails to remit payment in full within the stipulated timeframe of thirty (30) days from the date of invoice issuance, the Vendor reserves the explicit right to suspend all ongoing services and impose a late penalty fee of one and one-half percent (1.5%) per month on the outstanding balance. Furthermore, any subsequent legal costs incurred during the collection process shall be borne entirely by the Purchaser."

ex3_paragraph_simple = "If the buyer doesn't pay the full money in 30 days after getting the invoice, the seller can pause all services and charge a 1.5% monthly fee on the unpaid money. The buyer must pay for any legal fees needed to collect the money."

pipe = pipeline(
    "text-generation", 
    model="Qwen/Qwen2-0.5B-Instruct", 
    dtype=ms.float32  
)

print("CogniBridge is ready!\n")

def cognibridge_simplify(text):
    """Simplifies text (sentences or paragraphs) using Qwen with dynamic token limits."""
    
    # 1. Calculate dynamic tokens based on input length
    word_count = len(text.split())
    # Give it 2x the input size plus a small 20-token buffer. 
    # Cap it at 512 just in case you pass an entire book page.
    dynamic_max_tokens = min(int((word_count * 2) + 20), 512)
    
    prompt = f"""You are an expert at simplifying complex English. Look at these examples of complex text being rewritten into simple text.

Complex: {ex1_complex}
Simple: {ex1_simple}

Complex: {ex2_complex}
Simple: {ex2_simple}

Complex: {ex3_paragraph_complex}
Simple: {ex3_paragraph_simple}

Now, simplify this text using the same style. ONLY output the simplified text. Do not add any explanations, notes, or extra text.
Complex: {text}
Simple:"""

    result = pipe(
        prompt, 
        max_new_tokens=dynamic_max_tokens,  # <--- Using your brilliant dynamic limit!
        return_full_text=False, 
        temperature=0.1       
    )

    raw_output = result[0]["generated_text"].strip()
    
    # 2. The Final Failsafe
    # If the AI tries to start a new example by generating the word "Complex:", 
    # we immediately chop the string in half and throw away the gibberish.
    clean_output = raw_output.split("Complex:")[0].strip()
    
    return clean_output

def process_document(input_filename, output_filename):
    """Reads a text file line-by-line, simplifies it, and saves the output."""
    
    # 1. Safety check to make sure the file actually exists
    if not os.path.exists(input_filename):
        print(f"Error: I couldn't find '{input_filename}' in this folder.")
        return

    print(f"Opening '{input_filename}' for processing...\n")
    
    # 2. Open the input file to read, and the output file to write
    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        
        # 3. Read through the file one line at a time
        for line_number, line in enumerate(infile, 1):
            original_sentence = line.strip()
            
            # Skip empty lines so we don't waste the AI's time
            if not original_sentence:
                continue
                
            print(f"Simplifying sentence {line_number}...")
            
            # 4. Feed the sentence to our AI tool
            simplified = cognibridge_simplify(original_sentence)
            
            # 5. Write the results beautifully formatted into the output file
            outfile.write(f"Original:   {original_sentence}\n")
            outfile.write("-" * 50 + "\n\n")
            outfile.write(f"Simplified: {simplified}\n")
            
            
    print(f"\nSuccess! All simplified sentences have been saved to '{output_filename}'")


# --- How to run the batch processor ---
if __name__ == "__main__":
    # Create a simple text file named "complex_text.txt" in your project folder, 
    # paste some hard sentences in there (one per line), and run this script!
    
    process_document("../data/complex_text.txt", "../data/simplified_text.txt")
    pass