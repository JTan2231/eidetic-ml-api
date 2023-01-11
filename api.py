import torch
import numpy as np

from flask import Flask, jsonify, request
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModel

MAXLEN = 512

def normalize(v):
    """
    v.shape == (768,)
    """
    norm = np.linalg.norm(v)
    
    if norm == 0: 
       return v

    return v / norm

def split_pad(tensor):
    tensors = list(torch.split(tensor, MAXLEN, dim=1))

    shape = [x for x in tensors[-1].size()]
    shape[-1] = MAXLEN - tensors[-1].size(1)

    tensors[-1] = torch.cat([tensors[-1], torch.full(shape, 0)], axis=1)

    return torch.cat(tensors)

class LanguageModel:
    def __init__(self):
        self.initialized = False
    
    def initialize_models(self):
        self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-base')
        self.model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
        self.model.eval()
    
    def forward(self, input_text):
        inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt")

        split = inputs['input_ids'].size()[-1] > MAXLEN
        if split:
            inputs['input_ids'] = split_pad(inputs['input_ids'])
            inputs['attention_mask'] = split_pad(inputs['attention_mask'])

        outputs = self.model(**inputs)

        embed = outputs.pooler_output
        if split:
            embed = torch.mean(embed, 0, keepdim=True)

        embed = torch.squeeze(embed).detach().numpy()
        embed = normalize(embed)

        return embed

lm = LanguageModel()
lm.initialize_models()

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/get-embedding", methods=['POST'])
def get_embedding():
    if request.method == 'POST':
        data = request.json

        embedding = lm.forward(data['text'])
        embedding = [str(round(i, 4)) for i in embedding]

        return jsonify({ 'message': 'success', 'embedding': embedding })
    else:
        return jsonify({ 'message': "didn't work" })
