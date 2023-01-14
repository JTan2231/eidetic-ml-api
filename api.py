import os
import torch
import pathlib
import numpy as np
import subprocess as sp

from flask import Flask, jsonify, request
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModel

from colbert.modeling.colbert import ColBERT
from colbert.infra.config import ColBERTConfig
from colbert.modeling.tokenization.query_tokenization import QueryTokenizer
from colbert.modeling.tokenization.doc_tokenization import DocTokenizer
from colbert.utils.utils import load_checkpoint

MAXLEN = 300

#PATH = os.environ['HOME'] + '/.cloudnote/'
PATH = "/home/joey/.eidetic/"

CHECKPOINT = PATH + "colbertv2.0/"

# download weights if they're not here and we're not in AWS
if str(pathlib.Path().resolve())[:5] != '/var/' and not os.path.isdir(CHECKPOINT):
    print("Collecting ColBERT weights to", CHECKPOINT)

    sp.run(f"mkdir -p {PATH}".split(' '))
    sp.run(f"wget -O {PATH}colbert_weights.tar.gz https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz".split(' '))
    sp.run(f"tar -xvzf {PATH}colbert_weights.tar.gz -C {PATH}".split(' '))

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

        config = ColBERTConfig.load_from_checkpoint(CHECKPOINT)
        config.query_maxlen = MAXLEN
        config.doc_maxlen = MAXLEN
        config.nway = 1
        config.interaction = "colbert"

        self.query_tokenizer = QueryTokenizer(config)
        self.entry_tokenizer = DocTokenizer(config)

        self.model = ColBERT(CHECKPOINT, colbert_config=config)

    def detach(self, tensor):
        return torch.squeeze(tensor, dim=0).detach().numpy()
    
    def forward(self, input_text):
        tensors = self.entry_tokenizer.tensorize([input_text])
        embedding, mask = self.model.doc(*tensors, keep_dims='return_mask')

        embedding = normalize(self.detach(embedding))
        mask = self.detach(mask)

        return embedding, mask

lm = LanguageModel()

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/ping", methods=['GET'])
def ping():
    return jsonify({ 'message': 'alive' })

@app.route("/get-embedding", methods=['POST'])
def get_embedding():
    if request.method == 'POST':
        data = request.json

        embedding, mask = lm.forward(data['text'])
        print(embedding.shape)

        embedding = [[str(round(i, 5)) for i in v] for v in embedding]
        mask = [str(i[0]) for i in mask]

        return jsonify({ 'message': 'success', 'mask': mask, 'embedding': embedding })
    else:
        return jsonify({ 'message': "didn't work" })
