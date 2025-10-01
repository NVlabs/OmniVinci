from transformers import AutoProcessor, AutoModel, AutoConfig,AutoModelForCausalLM
import torch
import os

# default: Load the model on the available device(s)
model_path = "./"
audio_path = "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/AMI_ASR/ami/audio/ihm/train/EN2001a/train_ami_en2001a_h00_mee068_0000557_0000594.wav"
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
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": "Transcribe the whole speech."}
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