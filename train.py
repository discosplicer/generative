import os

import torch
import pandas as pd
import numpy as np

from data import ModelData
from model import Model, SimpleLinear, AttentionHead, BigramLanguageModel

# GPT-3 Mini
# 192 embeddings
# 6 heads
# 6 layers
# 1e-3 LR and 0.2 Dropout
# or 6e-4 LR and 0.1 Dropout

MODEL_PARAMS = {
    "batch_size": 32,  # Number of independent sequences.
    "block_size": 128,  # Max context length.
    "lr": 6e-4,
    "eval_iters": 100,
    "num_heads": 6,
    "n_layer": 6,
    "temperature": 0.8,
    "dropout": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
model_params = MODEL_PARAMS
MAX_ITERATIONS = 5000
INFERENCE_TOKENS = 500
# Coupon collector problem. With larger block sizes, each token needs to appear more
# often in the training data to give additional context.
n = model_params["block_size"]
MIN_PAIRS = n * np.log(n) + 0.5 * n

data = ModelData("based.txt", MIN_PAIRS, model_params["num_heads"], train_split_pct=0.9)
model_params["gate_tokens"] = data.tokenizer.gate_tokens
model_params["word_token"] = data.tokenizer.word_token
model_params["vocab_size"] = data.vocab_size
model_params["n_embd"] = data.vocab_size
model_params["head_size"] = model_params["n_embd"] // model_params["num_heads"]
model = Model(data, SimpleLinear, model_params)

model.train(data, MAX_ITERATIONS, INFERENCE_TOKENS)


# print(model.context_target_demo())
