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
Example script for video understanding using the model.

This script demonstrates how to:
1. Load the model and processor
2. Configure video and audio processing parameters
3. Process video input with optional audio
4. Generate description output

Usage:
    python example_mini_video.py --model_path <path_to_model> --video_path <path_to_video>
"""

from transformers import AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
import os
import argparse

# Configuration
parser = argparse.ArgumentParser(description="Video understanding example")
parser.add_argument("--model_path", type=str, default="./", help="Path to the model")
parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
parser.add_argument("--num_video_frames", type=int, default=128, help="Number of video frames to process")
parser.add_argument("--audio_length", type=str, default="max_3600", help="Maximum audio length")
parser.add_argument("--prompt", type=str, default="What are they talking about in detail?", help="Text prompt for the model")
parser.add_argument("--load_audio", action="store_true", default=True, help="Load audio from video")

args = parser.parse_args()

model_path = args.model_path
video_path = args.video_path
generation_kwargs = {"max_new_tokens": args.max_new_tokens, "max_length": 99999999}
load_audio_in_video = args.load_audio
num_video_frames = args.num_video_frames
audio_length = args.audio_length
text_prompt = args.prompt

assert os.path.exists(video_path), f"Video path {video_path} does not exist."

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

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
    """Run inference on video with text prompt."""
    print(f"Text prompt: {text_prompt}")
    print(f"Video path: {video_path}")
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": text_prompt}
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
