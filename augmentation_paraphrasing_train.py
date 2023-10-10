import os
import json
import copy
from transformers import *

directory_path_train = "Data/train/"

corpus_train = []

for filename in os.listdir(directory_path_train):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path_train, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            try:
                # Parse the string content as a dictionary
                json_data = json.loads(file_content)
                corpus_train.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {filename}: {e}")

model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = PegasusTokenizerFast.from_pretrained("tuner007/pegasus_paraphrase")

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)

new_corpus = copy.deepcopy(corpus_train)

for corpus in new_corpus:
    corpus['options'][0] = get_paraphrased_sentences(model, tokenizer, corpus['options'][0], num_beams=2, num_return_sequences=1)[0]
    corpus['options'][1] = get_paraphrased_sentences(model, tokenizer, corpus['options'][1], num_beams=2, num_return_sequences=1)[0]
    corpus['options'][2] = get_paraphrased_sentences(model, tokenizer, corpus['options'][2], num_beams=2, num_return_sequences=1)[0]
    corpus['options'][3] = get_paraphrased_sentences(model, tokenizer, corpus['options'][3], num_beams=2, num_return_sequences=1)[0]
    corpus['article'] = get_paraphrased_sentences(model, tokenizer, corpus['article'], num_beams=2, num_return_sequences=1)[0]

# Define the output directory where you want to save the .txt files
output_directory = "Data/train_para/"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for index, json_data in enumerate(new_corpus):
    # Create a unique filename for each item in the corpus (you can change this as needed)
    filename = f"augmented_{index}.txt"
    
    # Construct the full file path
    file_path = os.path.join(output_directory, filename)
    
    # Convert the JSON data back to a string
    json_str = json.dumps(json_data, indent=4)  # You can adjust the indentation as needed
    
    # Write the JSON string to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json_str)
