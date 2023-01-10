import torch

from flask import Flask, jsonify, request
from flask_cors import CORS

from transformers import BertTokenizer, BertModel

class LanguageModel:
    def __init__(self):
        self.initialized = False
    
    def initialize_models(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
    
    def forward(self, input_text):
        inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt")
        outputs = self.model(**inputs)

        embed = torch.squeeze(outputs.pooler_output).detach()
        
        return embed

lm = LanguageModel()
lm.initialize_models()

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/get-embeddings", methods=['POST'])
def get_embedding():
    if request.method == 'POST':
        data = request.json

        embedding = [str(round(i.item(), 4)) for i in lm.forward(data['text'])]

        print(len(', '.join(embedding).encode('utf-8')))

        return jsonify({ 'message': embedding })
    else:
        return jsonify({ 'message': "didn't work" })
