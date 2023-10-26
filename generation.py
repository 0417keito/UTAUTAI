import os
import torch
from einops import rearrange

from utautai.text2semantic.extract_t2s import generate_stok
from utautai.prompt_tts.style_module import get_style_module
from utautai.soundstorm import SoundStorm
from utautai.dataset.audio_processor import AudioTokenizer

style_module_ckpt_dir = ''
style_module_name = ''

soundstorm_ckpt_dir = ''
soundstorm_name = ''

def preload_model():
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    style_module = get_style_module()
    checkpoint = torch.load(os.path.join(style_module_ckpt_dir, style_module_name), map_location='cpu')
    missing_keys, unexpected_keys = style_module_ckpt_dir.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    style_module.eval()
    
    soundstorm = SoundStorm
    soundstorm.load(os.path.join(soundstorm_ckpt_dir, soundstorm_name))
    soundstorm.eval()
    
    codec = AudioTokenizer()
    return style_module, soundstorm, codec


def generate_audio(lyrics, prompt, steps=8, greedy=True, prompt_music=None):
    style_module, soundstorm, codec = preload_model()
    lyrics_semantic_token_ids = generate_stok(lyrics)
    music_semantic_token_ids = style_module(prompt)
    semantic_token_ids = torch.cat([lyrics_semantic_token_ids, music_semantic_token_ids],
                                   axis=1)
    if prompt_music is not None:
        prompt_tokens = codec.encode(prompt_music)
    generated = soundstorm.generate(semantic_tokens=semantic_token_ids,
                                    prompt_tokens=prompt_tokens, steps=steps, greedy=greedy)
    wavs = codec.decode(rearrange(generated, 'n q -> q b n')) #later
    
    return wavs