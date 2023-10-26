import argparse
import logging
import os
import numpy as np
from pathlib import Path
import langid
from vocos import Vocos

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
from .t2s_models.valle import VALLE
from .t2s_models.macros import *
from ..dataset.audio_processor import AudioTokenizer
from ..dataset.text_processor import TextProcessor, get_text_token_collater

checkpoints_dir = "./checkpoints/"

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

model_checkpoint_name = "vallex-checkpoint.pt"

def preload_models():
    if not os.path.exists(checkpoints_dir): os.mkdir(checkpoints_dir)
    # VALL-E
    model = VALLE(
        N_DIM,
        NUM_HEAD,
        NUM_LAYERS,
        norm_first=True,
        add_prenet=False,
        prefix_mode=PREFIX_MODE,
        share_embedding=True,
        nar_scale_factor=1.0,
        prepend_bos=True,
        num_quantizers=NUM_QUANTIZERS,
    ).to(device)
    checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.eval()
    
    return model

@torch.no_grad()
def generate_stok(text, language='auto'):
    model = preload_models()
    text_collater = get_text_token_collater
    text_processor = TextProcessor()
    
    
    text = text.replace("\n", "").strip(" ")
    # detect language
    if language == "auto":
        language = langid.classify(text)[0]
    lang_token = lang2token[language]
    lang = token2lang[lang_token]
    text = lang_token + text + lang_token
    
    audio_prompts = torch.zeros([1, 0, NUM_QUANTIZERS]).type(torch.int32).to(device)
    text_prompts = torch.zeros([1, 0]).type(torch.int32)

    logging.info(f"synthesize text: {text}")
    phone_tokens, langs = text_processor.tokenize(text=f"_{text}".strip())
    text_tokens, text_tokens_lens = text_collater(
        [
            phone_tokens
        ]
    )
    text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
    # accent control
    generated_stok = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        audio_prompts,
        top_k=-100,
        temperature=1,
    )

    return generated_stok