from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import normalize
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import pickle
import numpy as np
from sklearn.preprocessing import normalize


class Generate_Embeddings_Sequence:
    def __init__(self, data, model_name="dunzhang/stella_en_1.5B_v5", max_token_limit=512):
        self.max_token_limit = max_token_limit  # Set the maximum token limit per batch

        # Load the Hugging Face model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Accept pre-loaded data dictionary
        if isinstance(data, dict):
            self.data = data
        else:
            raise TypeError("The `data` parameter must be a dictionary containing the sequences.")

        # Generate embeddings specifically for sequence data
        self.emb = self.generate_embeddings(self.data, 'seq')

    def count_tokens(self, text):
        # Count tokens using the tokenizer
        return len(self.tokenizer.tokenize(text))

    def generate_embeddings_batch(self, texts):
        # Tokenize and generate embeddings, ensuring max token length is 512
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_token_limit)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Normalize embeddings to unit length
        return normalize(embeddings)

    def sequence_to_text(self, sequence):
        # Convert sequence data (NumPy array) to text format
        if isinstance(sequence, np.ndarray):
            return " ".join(map(str, sequence))
        return "[]"

    def generate_embeddings(self, data, cat):
        data_emb = {}
        batch_texts = []
        batch_ids = []
        batch_token_count = 0

        for id_adm, dat in tqdm(data.items()):
            # Convert sequence to text
            text = self.sequence_to_text(dat[cat])

            # Count tokens for the current text
            token_count = self.count_tokens(text)

            # Check if adding this text would exceed the token limit
            if batch_token_count + token_count > self.max_token_limit:
                if batch_texts:  # Ensure batch_texts is not empty
                    embeddings = self.generate_embeddings_batch(batch_texts)
                    for i, emb in enumerate(embeddings):
                        data_emb[batch_ids[i]] = {
                            'emb': emb,
                            'age': data[batch_ids[i]]['age'],
                            'label': data[batch_ids[i]]['label']
                        }
                    # Clear the batch and reset counters
                    batch_texts.clear()
                    batch_ids.clear()
                    batch_token_count = 0

            # Add text and ID to the batch
            batch_texts.append(text)
            batch_ids.append(id_adm)
            batch_token_count += token_count

        # Process the remaining batch
        if batch_texts:  # Ensure the remaining batch is not empty
            embeddings = self.generate_embeddings_batch(batch_texts)
            for i, emb in enumerate(embeddings):
                data_emb[batch_ids[i]] = {
                    'emb': emb,
                    'age': data[batch_ids[i]]['age'],
                    'label': data[batch_ids[i]]['label']
                }

        return data_emb

