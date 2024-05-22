import os
import re
import json
import time
import copy
import hashlib
import threading
from functools import wraps
from typing import Optional, Literal, Any
from dataclasses import field, dataclass, asdict

import torch
from transformers import TrainingArguments as TransformerTrainingArguments, TrainerCallback, HfArgumentParser


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DTYPES = {
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32
}


@dataclass
class ModelArguments:
    dtype: Optional[Literal["bf16", "f16", "f32"]] = "bf16"
    load_in_4bit: Optional[bool] = False
    load_in_8bit: Optional[bool] = False
    model_name_or_path: Optional[str] = "adept/fuyu-8b"
    platform: Optional[Literal["openai", "together", "huggingface"]] = "huggingface"




def format_messages(batch) :
    """Assumes batch is a list of lists of strings, where each inner list is a list of chat messages that alternate between user and assistant."""
    return [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": m
    } for i, m in enumerate(b)] for b in batch]




def throttle(lock: threading.Lock, rqi: int, last_requests, interval: int = 60) -> None:
    """Decorator to throttle the number of requests per interval.

    Args:
    lock: Lock to make checking the limit and sending requests thread-safe
    rqi: Requests per interval limit
    last_requests: List to store timestamps of the last requests
    interval: Interval in seconds to check the number of requests
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_requests
            while True:
                with lock:
                    # Remove timestamps older than 60 seconds
                    now = time.time()
                    last_requests = [req for req in last_requests if now - req < interval]
                    # If the number of requests in the last minute is less than the limit, send a new request
                    if len(last_requests) < rqi:
                        last_requests.append(now)
                        break
                    # Otherwise, wait for some time and try again
                    minimum = min([now - req for req in last_requests])
                time.sleep(minimum)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def catch_error_return_none(func):
    """Decorator to log a warning if an error occurs but catches the error and returns None."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred in {func.__name__}: {e}")
            return None
    return wrapper


defaults = {
    "beta": 0.1,
    "sigma_soft": 0.2,
    "sigma_hard": 0.1,
    "hard": 0.2,
    "soft": 0.1,
    "lr": 5e-5,
    "r": 32,
    "alpha": 64,
    "epochs": 1,
    "max_prompts": None,
    "multi_feedback_training": False,
    "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
    "use_base_prefix": None
}


