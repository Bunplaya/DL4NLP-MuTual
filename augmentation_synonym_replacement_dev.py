import os
import json
import nltk
from nltk.corpus import wordnet
import random

directory_path_train = "Data/dev/"

corpus_dev = []

for filename in os.listdir(directory_path_train):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path_train, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()
            try:
                # Parse the string content as a dictionary
                json_data = json.loads(file_content)
                corpus_dev.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON in file {filename}: {e}")

# Download WordNet data (run once)
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def synonym_replacement(text, n=1):
    words = text.split()
    augmented_text = []

    for word in words:
        if wordnet.synsets(word):
            synonyms = get_synonyms(word)
            if synonyms:
                random.shuffle(synonyms)
                synonym = synonyms[0]
                augmented_text.append(synonym)
            else:
                augmented_text.append(word)
        else:
            augmented_text.append(word)

    return ' '.join(augmented_text)

import copy
new_corpus = copy.deepcopy(corpus_dev)

for corpus in new_corpus:
    corpus['options'][0] = synonym_replacement(corpus['options'][0])
    corpus['options'][1] = synonym_replacement(corpus['options'][1])
    corpus['options'][2] = synonym_replacement(corpus['options'][2])
    corpus['options'][3] = synonym_replacement(corpus['options'][3])
    corpus['article'] = synonym_replacement(corpus['article'])

    import os
import json

# Define the output directory where you want to save the .txt files
output_directory = "Data/dev_syn/"

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
