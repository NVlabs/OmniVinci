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
Example script for image understanding using the model.

This script demonstrates how to:
1. Load the model and processor
2. Process image input
3. Generate description output

Usage:
    python example_mini_image.py --model_path <path_to_model> --image_path <path_to_image>
"""

from transformers import AutoProcessor, AutoModel, AutoConfig, AutoModelForCausalLM
import torch
import os
import argparse

# Configuration
parser = argparse.ArgumentParser(description="Image understanding example")
parser.add_argument("--model_path", type=str, default="./", help="Path to the model")
parser.add_argument("--image_path", type=str, required=True, help="Path to the image file")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
parser.add_argument("--prompt", type=str, default="Describe the image in detail.", help="Text prompt for the model")

args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path
generation_kwargs = {"max_new_tokens": args.max_new_tokens, "max_length": 99999999}

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

conversation = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": args.prompt}
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
