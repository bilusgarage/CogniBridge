import huggingface_hub
# Our trusty monkey patch
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import mindspore as ms
import mindnlp
from transformers import pipeline
from datasets import load_dataset

print("Initializing CogniBridge AI... (Loading dataset and model)")

# 1. Load the dataset and set up our examples ONCE at the start of the script
dataset = load_dataset("waboucay/wikilarge", 'original', split="train")

ex1_complex = dataset[0]['complex']
ex1_simple = dataset[0]['simple']

ex2_complex = dataset[1]['complex']
ex2_simple = dataset[1]['simple']

ex3_complex = dataset[2]['complex']
ex3_simple = dataset[2]['simple']

# 2. Load the Qwen model ONCE into your Mac's CPU memory
pipe = pipeline(
    "text-generation", 
    model="Qwen/Qwen2-0.5B-Instruct", 
    dtype=ms.float32  
)

print("CogniBridge is ready!\n")

# 3. Define your custom function
def cognibridge_simplify(sentence):
    """
    Takes a complex string, feeds it through Qwen with WikiLarge examples,
    and returns only the simplified sentence.
    """
    prompt = f"""You are an expert at simplifying complex English. Look at these examples of complex sentences being rewritten into simple sentences.

Complex: {ex1_complex}
Simple: {ex1_simple}

Complex: {ex2_complex}
Simple: {ex2_simple}

Complex: {ex3_complex}
Simple: {ex3_simple}

Now, simplify this sentence using the same style. ONLY output the simplified sentence. Do not add any explanations, notes, or extra text.
Complex: {sentence}
Simple:"""

    # Generate the text
    result = pipe(
        prompt, 
        max_new_tokens=50,      
        return_full_text=False, 
        temperature=0.1       
    )

    # Grab the raw text and aggressively chop off any AI "yapping"
    raw_output = result[0]["generated_text"].strip()
    clean_simplified_sentence = raw_output.split('\n')[0].strip()
    
    return clean_simplified_sentence

# --- Testing the function ---
if __name__ == "__main__":
    test_sentence_1 = "The utilization of this technological apparatus will fundamentally ameliorate the efficiency of our daily operational paradigms."
    test_sentence_2 = "The physician recommended that the patient consume an abundance of hydration to expedite the recuperation process."
    
    print("Test 1 Original:", test_sentence_1)
    print("Test 1 Simplified:", cognibridge_simplify(test_sentence_1))
    print("-" * 40)
    print("Test 2 Original:", test_sentence_2)
    print("Test 2 Simplified:", cognibridge_simplify(test_sentence_2))