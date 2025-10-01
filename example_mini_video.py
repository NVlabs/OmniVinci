from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM
import torch
import os

# default: Load the model on the available device(s)
model_path = "./"
# video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/tim.mp4"
# video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/trim_gpt5.mp4"
# video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/trim_gtc_1min.mp4"
# video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/ssvid.net--Introducing-NVIDIA-Jetson-AGX-Thor-The-ultimate-platform-for-physical_1080p.mp4"
# video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/jense_masayoshi.mp4"
# video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/trim_jensen_masayoshi.mp4"
video_path = "/lustre/fsw/portfolios/nvr/users/hanrongy/dataset/demo_videos/trim_jensen_interview.mp4"

generation_kwargs = {"max_new_tokens": 1024, "max_length": 99999999}
load_audio_in_video = True
num_video_frames = 128
audio_length = "max_3600"
# text_prompt = "Assess the video, followed by a detailed description of it's video and audio contents."
text_prompt="What are they talking about in detail?"
# text_prompt = "Transcribe the whole speech."
# text_prompt = "Transcribe the video."
# text_prompt = "Transcribe the audio."

assert os.path.exists(video_path), f"Video path {video_path} does not exist."
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

def forward_inference(video_path, text_prompt):
    print(f"text_prompt: {text_prompt}")
    print(f"video_path: {video_path}")
    conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": text_prompt}
                # {"type": "text", "text": "Transcribe the whole speech."}
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

forward_inference(video_path, text_prompt)
import pdb; pdb.set_trace()