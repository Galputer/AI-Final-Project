from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tqdm import tqdm
# Concatenated keywords
keywords = 'apple iphone technology'

# List of text strings
# Load DataFrame
df = pd.read_csv('book-csv/2005.csv')
# Load pre-trained BERT model and tokenizer
MODEL = 'bert-base-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

# df['encoded_text'] = df['clean_text'].apply(lambda x: tokenizer(x.lower(), padding=True, truncation=True, return_tensors='pt')['input_ids'])
# df['concatenated_text'] = df['text_tokens'].apply(lambda x: ' '.join(keywords_tokens + x))

# Encode keyword and text tokens using BERT
encoded_keywords = tokenizer(keywords.lower(), return_tensors='pt')['input_ids']
# encoded_texts = tokenizer(list(df['clean_text']), padding=True, truncation=True, return_tensors='pt')['input_ids']

batch_size = 64
total_batches = (len(df) + batch_size - 1) // batch_size

df['similarity_score'] = ''
df['encoded_text'] = ''
# Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
with torch.no_grad():
    keyword_embeddings = model(encoded_keywords.to(device))[0][:, 0, :].cpu().numpy()
    for i in tqdm(range(total_batches)):
        # Get batch start and end indices
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(df))
        
        # Encode texts in the batch
        batch_text_tokens = df.iloc[start_idx:end_idx]['clean_text']
        encoded_texts = tokenizer(batch_text_tokens.tolist(), padding=True, truncation=True, return_tensors='pt').to(device)
        df.iloc[start_idx:end_idx, df.columns.get_loc('encoded_text')] = encoded_texts
        # Generate BERT embeddings for the batch
        input_ids = encoded_texts['input_ids']
        attention_mask = encoded_texts['attention_mask']
        text_embeddings = model(input_ids.to(device), attention_mask=attention_mask)[0][:, 0, :].cpu().numpy()
        # print(text_embeddings)
        
        # Compute cosine similarity between keyword vector and text vectors in the batch
        similarity_scores = cosine_similarity(keyword_embeddings, text_embeddings)
        
        # Add similarity scores to DataFrame
        df.iloc[start_idx:end_idx, df.columns.get_loc('similarity_score')] = similarity_scores[0]

# Sort DataFrame by similarity score in descending order and show top 10 rows
top_10 = df.sort_values(by='similarity_score', ascending=False).head(10)
print(top_10)
top_10.to_csv("example.csv",index=False)
df.to_pickle("book-pickle/2005.csv",index=False)