# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script for audio transcription using the model.

This script demonstrates how to:
1. Load the model and processor
2. Configure audio processing parameters
3. Process audio input
4. Generate transcription output

Usage:
    python example_mini_audio.py --model_path <path_to_model> --audio_path <path_to_audio>
"""

from transformers import AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
import os
import argparse

# Configuration
parser = argparse.ArgumentParser(description="Audio transcription example")
parser.add_argument("--model_path", type=str, default="./", help="Path to the model")
parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
parser.add_argument("--num_video_frames", type=int, default=128, help="Number of video frames to process")
parser.add_argument("--audio_length", type=str, default="max_3600", help="Maximum audio length")

args = parser.parse_args()

model_path = args.model_path
audio_path = args.audio_path
generation_kwargs = {"max_new_tokens": args.max_new_tokens, "max_length": 99999999}
load_audio_in_video = True
num_video_frames = args.num_video_frames
audio_length = args.audio_length

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