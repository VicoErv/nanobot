"""Image generation tools."""

import asyncio
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


_PIPELINE: StableDiffusionXLPipeline | None = None
_PIPELINE_LOCK = asyncio.Lock()


def _load_pipeline(model_id: str) -> StableDiffusionXLPipeline:
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe


async def _get_pipeline(model_id: str) -> StableDiffusionXLPipeline:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    async with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE
        loop = asyncio.get_running_loop()
        _PIPELINE = await loop.run_in_executor(None, _load_pipeline, model_id)
        return _PIPELINE


class SdxlImageTool(Tool):
    """Generate an image using SDXL and send it to the user."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._model_id = os.environ.get("SDXL_MODEL_ID", "stabilityai/sdxl-turbo")

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "generate_image_sdxl"

    @property
    def description(self) -> str:
        return "Generate an image using SDXL and send it to the user."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Positive prompt describing the desired image",
                    "minLength": 1,
                }
            },
            "required": ["prompt"],
        }

    async def execute(self, prompt: str, **kwargs: object) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        if not self._send_callback:
            return "Error: Message sending not configured"

        seed = random.randint(0, 2**31 - 1)

        try:
            pipe = await _get_pipeline(self._model_id)
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                negative_prompt="low quality, worst quality, watermark",
                num_inference_steps=4,
                guidance_scale=0.0,
                generator=generator,
                width=1024,
                height=1024,
            )
            image = result.images[0]
        except Exception as e:
            logger.error(f"SDXL generation failed: {e}")
            return f"Error: SDXL generation failed: {e}"

        media_dir = Path.home() / ".nanobot" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        filename = f"sdxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = media_dir / filename
        image.save(file_path)

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=f"Here is your SDXL image. (seed={seed})",
            media=[str(file_path)],
        )

        try:
            await self._send_callback(msg)
            return f"Image generated and sent: {file_path}"
        except Exception as e:
            logger.error(f"Failed to send SDXL image: {e}")
            return f"Error: Failed to send image: {e}"
