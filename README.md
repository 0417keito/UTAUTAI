# UTAUTAI: Unrestricted Tune Automated Technology Artificial Interigence

## README

## 📖 Quick Index
* [🚀Model Architecture](#-modelarchitecture)
* [🤔What is UTAUTAI?](#-what_is_utautai)
* [🐍Method](#-method)
* [🧠TODO](#-todo)
* [🙏Appreciation](#-appreciation)
* [⭐️Show Your Support](#-show_your_support)
* [🙆Welcome Contributions](#-welcom_contributions)

## 🚀Model Architecture
![UTAUTAI main architecture](https://github.com/0417keito/UTAUTAI/blob/main/main_architecture.jpg)
🙇sorry for hand-draw

## 🤔What is UTAUTAI?
An open-source repository aimed at generating matching vocal and instrumental tracks from lyrics, similar to Suno AI's Chirp and Riffusion.

## 🐍Method
UTAUTAI's method are mainly inspired by [SPEAR TTS](https://arxiv.org/abs/2302.03540)

During training, the input consists of semantic tokens obtained from 'lyrics2semantic AR', which extracts semantic tokens from lyrics, as well as Acoustic tokens. Additionally, [MERT](https://arxiv.org/abs/2306.00107) representations derived from the music are subjected to k-means quantization to obtain further semantic tokens.

However, during inference, it is not possible to obtain MERT representations from the music. Therefore, we train a Style Module following the methodology of [Prompt TTS2](https://arxiv.org/abs/2309.02285) to acquire the target MERT representations from the prompt during inference. The Style Module is composed of a transformer-based diffusion model.

I think that using this approach, we can successfully accomplish the target tasks. What do you think?

## 🧠TODO
- [ ] How can we obtain lyrics that match the cropped audio? Or should we even crop the audio in the first place? [code](https://github.com/0417keito/UTAUTAI/blob/main/utautai/dataset/labels.py#L12C5-L25C5)
- [ ] Examine the handling of phonemization and special tokens, and make necessary code modifications. [code](https://github.com/0417keito/UTAUTAI/blob/main/utautai/dataset/text_processor.py)
- [ ] Correct the collator in the dataset. [code](https://github.com/0417keito/UTAUTAI/blob/main/utautai/dataset/collate.py)
- [ ] Complete the StyleModule inference code. [code](https://github.com/0417keito/UTAUTAI/blob/main/utautai/prompt_tts/style_module.py#L51)
- [ ] Other minor code fixes, such as masking strategies.

## 🙏Appreciation
- [SPEAR TTS paper](https://arxiv.org/abs/2302.03540)
- [VALL-E paper](https://arxiv.org/abs/2301.02111)
- [JukeBox paper](https://arxiv.org/abs/2005.00341)
- [SoundStorm paper](https://arxiv.org/abs/2305.09636)
- [MusicLM paper](https://arxiv.org/abs/2301.11325)
- [AudioLM paper](https://arxiv.org/abs/2209.03143)
- [MusicGen paper](https://arxiv.org/abs/2306.05284)
- [PromptTTS2 paper](https://arxiv.org/abs/2309.02285)
- [lucidrains' Soundstorm repo](https://github.com/lucidrains/soundstorm-pytorch)
- [soundstorm speechtokenizer](https://github.com/ZhangXInFD/soundstorm-speechtokenizer)
- [lifeiteng's vall-e](https://github.com/lifeiteng/vall-e)
- [Plachtaa's VALL-E-X](https://github.com/Plachtaa/VALL-E-X)
- [bark](https://github.com/suno-ai/bark)

## ⭐️Show Your Support

If you find UTAUTAI interesting and useful, give us a star on GitHub! ⭐️ It encourages us to keep improving the model and adding exciting features.

## 🙆Welcome Contributions
Contributions are always welcome.
