import requests
from bs4 import BeautifulSoup
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import ssl
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

import sys
sys.path.append('KeyClass/keyclass/')
sys.path.append('KeyClass/scripts/')

import gc
import argparse
import label_data, encode_datasets, train_downstream_model
import torch
import pickle
import numpy as np
import os
from os.path import join, exists
from datetime import datetime
import utils
import models
import create_lfs
import train_classifier
import importlib

random_seed = 0 # Random seed for experiments

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('punkt_tab')

# Build our class descriptions table.

# Scrape codes from wikipedia.
icd_9_codes_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_ICD-9_codes_E_and_V_codes:_external_causes_of_injury_and_supplemental_classification")
soup = BeautifulSoup(icd_9_codes_wiki.text, "html.parser")

tds_with_nowrap = soup.find_all('td')

code_mapping = dict()
category_descs_concat = dict()

for td in tds_with_nowrap:
    span = td.find('span', class_='nowrap')
    if span:
        try:
            split = span.a['title'].split(":")
            title = split[1][1:]
            codes = split[0]
            codes = codes.split(" ")[4].split("–")
            for i in range(int(codes[0]), int(codes[1]) + 1):
                code_mapping[i] = title
            category_descs_concat[title] = ""
        except:
            pass

external_str = "external causes of injury"
supp_str = "supplementary"
category_descs_concat[external_str] = ""
category_descs_concat[supp_str] = ""

#Descriptions are pulled from cms.gov
with open('data/CMS32_DESC_LONG_DX.txt', 'r', encoding='latin1') as file:
    lines = [line.strip() for line in file]

for line in lines:
    split = line.split(" ")
    code = split[0][:3]
    desc = " ".join(split[2:])
    if code[0] == "E":
        category = external_str
    elif code[0] == "V":
        category = supp_str
    else:
        category = code_mapping[int(code)]
    category_descs_concat[category] += desc + " "

stop_words = set(stopwords.words('english'))
final_output = dict()
num_most_common = 20


for category in category_descs_concat:
    concat = category_descs_concat[category]
    tokenized = words = word_tokenize(concat.lower())
    tokenized = [word for word in tokenized if word not in stop_words and word not in string.punctuation]
    word_freq = Counter(tokenized)
    most_common = word_freq.most_common(num_most_common)
    most_common_lst = [word for word, _ in most_common]
    most_common_str = ' '.join(most_common_lst)
    final_output[category] = most_common_str

df = pd.DataFrame.from_dict(final_output, orient="index")

if not os.path.exists("data"):
    os.mkdir("data")
df.to_csv("data/mined_descriptions.csv")

# Generate targets for the config file
count = 0
for index, row in df.iterrows():
    if count < 10:
        target_name = "target_" + "0" + str(count)
    else:
        target_name = "target_" + str(count)
    val_str = row[0].replace(' ', ', ')
    print(target_name + ": " + val_str)
    count += 1

# Read the MIMIC-III data
icd = pd.read_csv('data/DIAGNOSES_ICD.csv')
notes = pd.read_csv('data/NOTEEVENTS.csv')

merged = pd.merge(notes, icd, on=["SUBJECT_ID", "HADM_ID"])
merged = merged.dropna(subset=['TEXT', 'ICD9_CODE'])
merged = merged[merged['CATEGORY'] == 'Discharge summary']

category_to_number_mapping = {}
count = 0
for key in final_output:
    category_to_number_mapping[key] = count
    count += 1

def map_labels(icd9_code):
    code = icd9_code[:3]
    if code[0] == "E":
        category = external_str
    elif code[0] == "V":
        category = supp_str
    else:
        category = code_mapping[int(code)]
    return category_to_number_mapping[category]

merged['label'] = merged['ICD9_CODE'].apply(map_labels)

data = merged[['TEXT', 'label']]
data = data.drop_duplicates(subset=['TEXT', 'label'])

# Encode the data
encoded_df_dict = {}
for index, row in data.iterrows():
    if row['TEXT'] not in encoded_df_dict:
        encoded_df_dict[row['TEXT']] = [row['label']]
    else:
        encoded_df_dict[row['TEXT']].append(row['label'])

for key in encoded_df_dict:
    val = encoded_df_dict[key]
    encoded_df_dict[key] = ' '.join(str(x) for x in encoded_df_dict[key])
encoded_df = pd.DataFrame.from_dict(encoded_df_dict, orient='index').reset_index()
encoded_df.columns = ['text', 'labels']

# Use TF-IDF to get the most relevant words in the text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sampled_df['text'])

feature_names = np.array(vectorizer.get_feature_names())
top_k = 50
new_texts = []

for doc_idx in range(tfidf_matrix.shape[0]):
    row = tfidf_matrix.getrow(doc_idx)
    row_data = row.data
    row_indices = row.indices
    
    # Sort indices by TF-IDF score
    top_indices = row_indices[np.argsort(row_data)[::-1][:top_k]]
    
    # Get corresponding words
    top_words = feature_names[top_indices]
    
    # Join them back into a document
    new_texts.append(' '.join(top_words))

# 3. Replace or add a new column
sampled_df['text'] = new_texts

# Write the table into train/split .txt files.
train_data, test_data = train_test_split(sampled_df, test_size=0.3, random_state=random_seed)

if not os.path.exists("data/mimic"):
    os.mkdir("data/mimic")

with open('data/mimic/train.txt', 'w', encoding='utf-8') as f_text, \
     open('data/mimic/train_labels.txt', 'w', encoding='utf-8') as f_label:
    for text, label in zip(train_data['text'], train_data['labels']):
        f_text.write(text.strip().replace('\n', ' ') + '\n')
        f_label.write(str(label) + '\n')

with open('data/mimic/test.txt', 'w', encoding='utf-8') as f_text, \
     open('data/mimic/test_labels.txt', 'w', encoding='utf-8') as f_label:
    for text, label in zip(test_data['text'], test_data['labels']):
        f_text.write(text.strip().replace('\n', ' ') + '\n')
        f_label.write(str(label) + '\n')

# Input arguments
config_file_path = r'mimic_config.yml' # Specify path to the configuration file
default_config_path = r'KeyClass/config_files/default_config.yml'

# Encode the dataset
args = utils.Parser(config_file_path=config_file_path, default_config_file_path=default_config_path).parse()

if args['use_custom_encoder']:
    model = models.CustomEncoder(pretrained_model_name_or_path=args['base_encoder'], 
        device='cuda' if torch.cuda.is_available() else 'cpu')
else:
    model = models.Encoder(model_name=args['base_encoder'], 
        device='cuda' if torch.cuda.is_available() else 'cpu')

for split in ['train', 'test']:
    sentences = utils.fetch_data(dataset=args['dataset'], split=split, path=args['data_path'])
    embeddings = model.encode(sentences=sentences, batch_size=args['end_model_batch_size'], 
                                show_progress_bar=args['show_progress_bar'], 
                                normalize_embeddings=args['normalize_embeddings'])
    with open(join(args['data_path'], args['dataset'], f'{split}_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)

# Probabilistically label the data

args = utils.Parser(config_file_path=config_file_path, default_config_file_path=default_config_path).parse()

# Load training data
train_text = utils.fetch_data(dataset=args['dataset'], path=args['data_path'], split='train')

training_labels_present = False
if exists(join(args['data_path'], args['dataset'], 'train_labels.txt')):
    with open(join(args['data_path'], args['dataset'], 'train_labels.txt'), 'r') as f:
        y_train = f.readlines()
    # y_train = np.array([int(i.replace('\n','')) for i in y_train])
    y_train = np.array([[int(i) for i in labels.strip().split()] for labels in y_train], dtype=object)
    training_labels_present = True
else:
    y_train = None
    training_labels_present = False
    print('No training labels found!')

with open(join(args['data_path'], args['dataset'], 'train_embeddings.pkl'), 'rb') as f:
    X_train = pickle.load(f)

# Change the target to 19-dimensional one-hot vectors as stated in the paper
binarizer = MultiLabelBinarizer()
y_train_encoded = binarizer.fit_transform(y_train)

# Print dataset statistics
print(f"Getting labels for the {args['dataset']} data...")
print(f'Size of the data: {len(train_text)}')
if training_labels_present:
    print('Class distribution', np.unique(np.hstack(y_train.flatten()), return_counts=True))

class_balance = np.unique(np.hstack(y_train), return_counts=True)[1] / np.unique(np.hstack(y_train), return_counts=True)[1].sum()

# Creating labeling functions

importlib.reload(create_lfs)
args = utils.Parser(config_file_path=config_file_path, default_config_file_path=default_config_path).parse()

# Load label names/descriptions
label_names = []
for a in args:
    if 'target' in a: label_names.append(args[a])

labeler = create_lfs.CreateLabellingFunctions(base_encoder=args['base_encoder'], 
                                            device=torch.device(args['device']),
                                            label_model=args['label_model'])

print("Training label model, this may take a while...")

proba_preds = labeler.get_labels(text_corpus=train_text, label_names=label_names, min_df=args['min_df'], 
                                ngram_range=args['ngram_range'], topk=args['topk'], y_train=y_train_encoded, 
                                label_model_lr=args['label_model_lr'], label_model_n_epochs=args['label_model_n_epochs'], 
                                verbose=True, n_classes=args['n_classes'], class_balance=class_balance)


# Save the predictions
if not os.path.exists(args['preds_path']): os.makedirs(args['preds_path'])
with open(join(args['preds_path'], f"{args['label_model']}_proba_preds.pkl"), 'wb') as f:
    pickle.dump(proba_preds, f)

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Load data
X_train_embed_masked, y_train_lm_masked, y_train_masked, \
    X_test_embed, y_test, training_labels_present, \
    sample_weights_masked, proba_preds_masked = train_downstream_model.load_data(args, class_balance=class_balance)

# Train a downstream classifier
importlib.reload(train_classifier)
importlib.reload(train_downstream_model)

args = utils.Parser(config_file_path=config_file_path, default_config_file_path=default_config_path).parse()

if args['use_custom_encoder']:
    encoder = models.CustomEncoder(pretrained_model_name_or_path=args['base_encoder'], device=args['device'])
else:
    encoder = models.Encoder(model_name=args['base_encoder'], device=args['device'])

classifier = models.FeedForwardFlexible(encoder_model=encoder,
                                        h_sizes=args['h_sizes'], 
                                        activation=eval(args['activation']),
                                        device=torch.device(args['device']))
print('\n===== Training the downstream classifier =====\n')
model = train_classifier.train(model=classifier, 
                            device=torch.device(args['device']),
                            X_train=X_train_embed_masked, 
                            y_train=y_train_lm_masked,
                            sample_weights=sample_weights_masked if args['use_noise_aware_loss'] else None, 
                            epochs=args['end_model_epochs'], 
                            batch_size=args['end_model_batch_size'], 
                            criterion=eval(args['criterion']), 
                            raw_text=False, 
                            lr=eval(args['end_model_lr']), 
                            weight_decay=eval(args['end_model_weight_decay']),
                            patience=args['end_model_patience'])


end_model_preds_train = model.predict_proba(torch.from_numpy(X_train_embed_masked), batch_size=512, raw_text=False)
end_model_preds_test = model.predict_proba(torch.from_numpy(X_test_embed), batch_size=512, raw_text=False)

# Self train the classifier
importlib.reload(train_classifier)

args = utils.Parser(config_file_path=config_file_path, default_config_file_path=default_config_path).parse()

with open(
        join(args['data_path'], args['dataset'], f'train_embeddings.pkl'),
        'rb') as f:
    X_train_embed = pickle.load(f)
with open(join(args['data_path'], args['dataset'], f'test_embeddings.pkl'),
          'rb') as f:
    X_test_embed = pickle.load(f)

model = train_classifier.self_train(model=model, 
                                    X_train=X_train_embed, 
                                    X_val=X_test_embed, 
                                    y_val=y_test, 
                                    device=torch.device(args['device']), 
                                    lr=eval(args['self_train_lr']), 
                                    weight_decay=eval(args['self_train_weight_decay']),
                                    patience=args['self_train_patience'], 
                                    batch_size=args['self_train_batch_size'], 
                                    q_update_interval=args['q_update_interval'],
                                    self_train_thresh=eval(args['self_train_thresh']), 
                                    print_eval=True, raw_text=False)

end_model_preds_test = model.predict_proba(torch.from_numpy(X_test_embed), batch_size=args['self_train_batch_size'], raw_text=False)

y_test_encoded = binarizer.fit_transform(y_test)

testing_metrics = utils.compute_metrics_bootstrap(y_preds=(end_model_preds_test>0.46).astype(int),
                                                    y_true=y_test_encoded, 
                                                    average=args['average'], 
                                                    n_bootstrap=args['n_bootstrap'], 
                                                    n_jobs=args['n_jobs'])

for i in range(len(testing_metrics)):
    if i == 0:
        print("Accuracy (mean, std): ", testing_metrics[i])
    elif i == 1:
        print("Precision (mean, std): ", testing_metrics[i])
    elif i == 2:
        print("Recall (mean, std): ", testing_metrics[i])
    else:
        print("F1 Score (mean, std): ", testing_metrics[i])