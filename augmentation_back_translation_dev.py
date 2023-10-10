import os
import json
from transformers import MarianMTModel, MarianTokenizer
import copy

directory_path_dev = "Data/dev/"

corpus_dev = []

for filename in os.listdir(directory_path_dev):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path_dev, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            try:
                # Parse the string content as a dictionary
                json_data = json.loads(file_content)
                corpus_dev.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {filename}: {e}")

# Get the name of the first model
first_model_name = 'Helsinki-NLP/opus-mt-en-fr'

# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)

# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name)

# Get the name of the second model
second_model_name = 'Helsinki-NLP/opus-mt-fr-en'

# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)

# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name)

def format_batch_texts(language_code, batch_texts):
  
  formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

  return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr"):
  # Prepare the text data into appropriate format for the model
  formated_batch_texts = format_batch_texts(language, batch_texts)
  
  # Generate translation using model
  translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True))

  # Convert the generated tokens indices back into text
  translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
  
  return translated_texts

def perform_back_translation(batch_texts, original_language="en", temporary_language="fr"):

  # Translate from Original to Tempora ry Language
  tmp_translated_batch = perform_translation(batch_texts, first_model, first_model_tkn, temporary_language)

  # Translate Back to English
  back_translated_batch = perform_translation(tmp_translated_batch, second_model, second_model_tkn, original_language)

  # Return The Final Result
  return back_translated_batch

def combine_texts(original_texts, back_translated_batch):
  
  return set(original_texts + back_translated_batch) 

def perform_back_translation_with_augmentation(batch_texts, original_language="en", temporary_language="fr"):

  # Translate from Original to Temporary Language
  tmp_translated_batch = perform_translation(batch_texts, first_model, first_model_tkn, temporary_language)

  # Translate Back to English
  back_translated_batch = perform_translation(tmp_translated_batch, second_model, second_model_tkn, original_language)

  # Return The Final Result
  # return combine_texts(original_texts, back_translated_batch)
  return back_translated_batch

new_corpus = copy.deepcopy(corpus_dev)

for corpus in new_corpus:
    corpus['options'][0] = perform_back_translation_with_augmentation({corpus['options'][0]})
    corpus['options'][1] = perform_back_translation_with_augmentation({corpus['options'][1]})
    corpus['options'][2] = perform_back_translation_with_augmentation({corpus['options'][2]})
    corpus['options'][3] = perform_back_translation_with_augmentation({corpus['options'][3]})
    corpus['article'] = perform_back_translation_with_augmentation({corpus['article']})

# Define the output directory where you want to save the .txt files
output_directory = "Data/dev_para/"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

for index, json_data in enumerate(corpus):
#for index, json_data in enumerate(new_corpus):
    # Create a unique filename for each item in the corpus (you can change this as needed)
    filename = f"augmented_{index}.txt"
    
    # Construct the full file path
    file_path = os.path.join(output_directory, filename)
    
    # Convert the JSON data back to a string
    json_str = json.dumps(json_data, indent=4)  # You can adjust the indentation as needed
    
    # Write the JSON string to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json_str)