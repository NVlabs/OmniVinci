<p align="center" width="100%">
<img src="assets/logo.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# <span style="background: linear-gradient(45deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: bold; font-size: 1.1em;">**OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM [[<u>Link</u>](https://arxiv.org/)]**</span> <br />

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/)
[![Code](https://img.shields.io/badge/GitHub-Link-blue)](https://github.com/NVlabs)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/nvidia/omnivinci)

<div align="center">

</div>

[Hanrong Ye*†](https://sites.google.com/site/yhrspace/home), Huck Yang†, Arushi Goel†, Wei Huang†, Ligeng Zhu†, Yuanhang Su†, Sean Lin†, An-Chieh Cheng†, Zhen Wan†, Jinchuan Tian†, Yuming Lou†, Dong Yang†, Zhijian Liu, Yukang Chen, Ambrish Dantrey, Ehsan Jahangiri, Sreyan Ghosh, Daguang Xu, Ehsan Hosseini Asl, Danial Mohseni Taheri, Vidya Murali, Sifei Liu, Jason Lu, Oluwatobi Olabiyi, Frank Wang, Rafael Valle, Bryan Catanzaro, Andrew Tao, Song Han, Jan Kautz, Hongxu Yin^†, Pavlo Molchanov^  
<span style="color: rgb(133, 184, 55);">**NVIDIA**</span>  
*Corresponding Author | †Core Authors | ^Senior Authors 

<p align="center" width="100%">
<img src="assets/performance.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world.
We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM.
We carefully study the design choices across model architecture and data curation.
For model architecture, we present three key innovations:
**(i)** OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space;
**(ii)** Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and
**(iii)** Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. 
We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, \modelname, improves over Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6$\times$ reduction compared to Qwen2.5-Omni’s 1.2T.
We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory. 

| Model        | Omni - Dailyomni | Omni - Worldsense | Audio - MMAU | Audio - MMAR | Vision - MVBench | Vision - Video-MME (w/o sub) |
|--------------|------------------|-------------------|--------------------------|--------------|------------------|------------------------------|
| Qwen2.5-Omni | 47.45            | 45.4              | 71.0                       | 56.7         | 70.3             | 64.3                         |
| Ours         | 66.5             | 48.23             | 71.6                     | 58.4         | 70.6             | 68.2                         |


## News
- [x] [2025.9.30] **OmniVinci-9B** is released! It supports joint understanding of **vision, audio, and text**.

## Model Usage

### Inference
### Envirnoment setup


1. Download and cd huggingface repo
```
huggingface-cli download nvidia/omnivinci --local-dir ./omnivinci --local-dir-use-symlinks False
cd ./omnivinci
```

2. Install python environment (based on NVILA codebase)
```
bash ./environment_setup.sh omnivinci
```

### 🤗 Transformers Usage

#### Video (with audio) Inference Example:
```python
from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM
import torch
import os

# default: Load the model on the available device(s)
model_path = "./"
video_path = "xxx.mp4"
generation_kwargs = {"max_new_tokens": 1024, "max_length": 99999999}
load_audio_in_video = True
num_video_frames = 128
audio_length = "max_3600"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(model_path,
                                  trust_remote_code=True,
                                  torch_dtype="torch.float16",
                                  device_map="auto")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
generation_config = model.default_generation_config
generation_config.update(**generation_kwargs)

model.config.load_audio_in_video = load_audio_in_video
processor.config.load_audio_in_video = load_audio_in_video
if num_video_frames > 0:
    model.config.num_video_frames = num_video_frames
    processor.config.num_video_frames = num_video_frames
if audio_length != -1:
    model.config.audio_chunk_length = audio_length
    processor.config.audio_chunk_length = audio_length


conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "video":video_path},
            {"type": "text", "text": "Assess the video, followed by a detailed description of it's video and audio contents."}
        ]
}]
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

inputs = processor([text])

output_ids = model.generate(
    input_ids=inputs.input_ids,
    media=getattr(inputs, 'media', None),
    media_config=getattr(inputs, 'media_config', None),
    generation_config=generation_config,
)
print(processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True))
```

- **For a audio and image inference examples please refer to ```example_mini_audio.py``` and ```example_mini_image.py```**


## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.

```bibtex
```bibtex
@inproceedings{omnivinci2025,
      title={OmniVinci},
      author={xxx},
      booktitle={xxx},
      year={2025},
}
```