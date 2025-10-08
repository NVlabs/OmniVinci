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


from transformers import AutoProcessor, AutoModel, AutoConfig, GenerationConfig
import torch
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import sys
os.environ["HF_HUB_OFFLINE"] = "1"  # Use local cache for models

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_to_sys_path_direct(model_path):
    """Add model path directly to sys.path"""
    if model_path not in sys.path:
        sys.path.insert(0, model_path)  # Insert at beginning for priority
        print(f"✓ Added to sys.path: {model_path}")
    else:
        print(f"Already in sys.path: {model_path}")

class NVOmniVideoInference:
    """A class to handle NVOmni video model inference with improved error handling and flexibility."""
    
    def __init__(self, model_path: str, torch_dtype="torch.float16", device_map="auto"):
        """
        Initialize the NVOmni model for video inference.
        
        Args:
            model_path (str): Path to the model directory
            torch_dtype: PyTorch data type for model weights
            device_map (str): Device mapping strategy for model loading
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.model = None
        self.processor = None
        self.config = None
        self.device = None
        
        self.load_model()
        
    def validate_paths(self, model_path: str, video_path: str = None) -> bool:
        """Validate that required paths exist."""
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False
            
        if video_path and not Path(video_path).exists():
            logger.error(f"Video path does not exist: {video_path}")
            return False
            
        return True
    
    def load_model(self) -> bool:
        """Load the model, processor, and config with error handling."""
        if not self.validate_paths(self.model_path):
            return False
            
        if True:
            logger.info("Loading model configuration...")
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            
            logger.info("Loading model...")
            start_time = time.time()
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                low_cpu_mem_usage=True  # More memory efficient loading
            )#.to(eval(self.torch_dtype))
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)

            # Set device for single-device setups
            if hasattr(self.model, 'device'):
                self.device = self.model.device
            else:
                self.device = next(self.model.parameters()).device if self.model.parameters() else torch.device('cpu')
            
            logger.info(f"Model successfully loaded on device: {self.device}")
            self._print_model_info()
            return True
            
    def _print_model_info(self):
        """Print useful information about the loaded model."""
        logger.info("=" * 50)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 50)
        
        if self.config:
            logger.info(f"Model type: {getattr(self.config, 'model_type', 'Unknown')}")
            logger.info(f"Hidden size: {getattr(self.config, 'hidden_size', 'Unknown')}")
            
        if self.model and torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    def create_conversation(self, video_path: str, text_prompt: str) -> List[Dict[str, Any]]:
        """
        Create a conversation format for the model.
        
        Args:
            video_path (str): Path to the video file
            text_prompt (str): Text prompt for the model
            
        Returns:
            List[Dict]: Conversation in the expected format
        """
        return [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": text_prompt}
            ]
        }]

    @torch.inference_mode()    
    def generate_response(
        self, 
        video_path: str, 
        text_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = None,
        num_video_frames: int = -1,
        load_audio_in_video: bool = True,
        audio_length: Union[int, str] = "max_3600",
    ) -> Optional[str]:
        """
        Generate a response from the model given a video and text prompt.
        
        Args:
            video_path (str): Path to the video file
            text_prompt (str): Text prompt for the model
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            do_sample (bool): Whether to use sampling
            custom_generation_config (GenerationConfig): Custom generation configuration
            
        Returns:
            Optional[str]: Generated response or None if failed
        """
        if not self.model or not self.processor:
            logger.error("Model or processor not loaded. Please initialize the model first.")
            return None
            
        if not self.validate_paths(self.model_path, video_path):
            return None
        
        # try:
        if True:
        
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Text prompt: {text_prompt}")
            
            # Create conversation
            conversation = self.create_conversation(video_path, text_prompt)
            
            # Apply chat template
            text = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            logger.info(f"Chat template applied")

            # set model params
            self.model.config.load_audio_in_video = load_audio_in_video
            self.processor.config.load_audio_in_video = load_audio_in_video
            if num_video_frames > 0:
                self.model.config.num_video_frames = num_video_frames
                self.processor.config.num_video_frames = num_video_frames
            if audio_length != -1:
                self.model.config.audio_chunk_length = audio_length
                self.processor.config.audio_chunk_length = audio_length
            logger.info(f"Model config - load_audio_in_video: {self.model.config.load_audio_in_video}, num_video_frames: {self.model.config.num_video_frames}, audio_chunk_length: {self.model.config.audio_chunk_length}")
            
            # Process inputs
            start_time = time.time()
            inputs = self.processor([text])
            
            # Move inputs to the correct device if needed
            if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                inputs.input_ids = inputs.input_ids.to(self.device)
            
            processing_time = time.time() - start_time
            logger.info(f"Input processing completed in {processing_time:.2f} seconds")
            
            logger.info("Generating response...")
            start_time = time.time()

            generation_kwargs = {"max_new_tokens": max_new_tokens, "max_length": 99999999}
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            if do_sample is not None:
                generation_kwargs["do_sample"] = do_sample
            if temperature is not None:
                generation_kwargs["temperature"] = temperature

            generation_config = self.model.default_generation_config
            generation_config.update(**generation_kwargs)

            logger.info(f"Generation config: {generation_config.to_dict()}")


            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    media=getattr(inputs, 'media', None),
                    media_config=getattr(inputs, 'media_config', None),
                    generation_config=generation_config,
                )
            
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.2f} seconds")
            
            # Decode response
            response = self.processor.tokenizer.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0]

            return response
            
    def batch_generate(
        self, 
        video_text_pairs: List[tuple], 
        **generation_kwargs
    ) -> List[Optional[str]]:
        """
        Generate responses for multiple video-text pairs.
        
        Args:
            video_text_pairs (List[tuple]): List of (video_path, text_prompt) tuples
            **generation_kwargs: Arguments passed to generate_response
            
        Returns:
            List[Optional[str]]: List of generated responses
        """
        responses = []
        for i, (video_path, text_prompt) in enumerate(video_text_pairs):
            logger.info(f"Processing batch item {i+1}/{len(video_text_pairs)}")
            response = self.generate_response(video_path, text_prompt, **generation_kwargs)
            responses.append(response)
            
            # Clear cache between generations to manage memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return responses

def main():
    """Main function demonstrating usage of the NVOmni model."""
    
    # Configuration
    MODEL_PATH = "./"
    VIDEO_PATH = "xxx.mp4"
    TEXT_PROMPT = "Assess the video, followed by a detailed description of it's video and audio contents."

    num_video_frames=128
    audio_length="max_3600"
    load_audio_in_video=True

    add_to_sys_path_direct(MODEL_PATH)
    
    # Initialize the inference class
    logger.info("Initializing NVOmni Video Inference...")
    inferencer = NVOmniVideoInference(MODEL_PATH, torch_dtype="torch.float16")
    
    if inferencer.model is None:
        logger.error("Failed to initialize model. Exiting.")
        return
    
    # Generate response
    logger.info("Starting inference...")
    response = inferencer.generate_response(
        video_path=VIDEO_PATH,
        text_prompt=TEXT_PROMPT,
        num_video_frames=num_video_frames,
        load_audio_in_video=load_audio_in_video,
        audio_length=audio_length,
        max_new_tokens=1024,
    )
    
    if response:
        print("\n" + "="*60)
        print("GENERATED RESPONSE")
        print("="*60)
        print(response)
        print("="*60)
    else:
        logger.error("Failed to generate response")
    
    # Example of batch processing
    if False:
        logger.info("\nExample: Batch processing")
        batch_pairs = [
            (VIDEO_PATH, "What is happening in this video?"),
            (VIDEO_PATH, "Describe the audio content of this video."),
        ]
        
        batch_responses = inferencer.batch_generate(batch_pairs, max_new_tokens=128)
        
        for i, (pair, response) in enumerate(zip(batch_pairs, batch_responses)):
            print(f"\n--- Batch Response {i+1} ---")
            print(f"Prompt: {pair[1]}")
            print(f"Response: {response}")

if __name__ == "__main__":
    main()