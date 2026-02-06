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
from PIL import Image

try:
    from diffusers import Flux2KleinPipeline
except Exception:  # pragma: no cover - optional dependency version
    Flux2KleinPipeline = None  # type: ignore[assignment]

from nanobot.agent.tools.base import Tool
from nanobot.bus.events import OutboundMessage


_PIPELINE: StableDiffusionXLPipeline | None = None
_PIPELINE_LOCK = asyncio.Lock()
_FLUX_PIPELINE: "Flux2KleinPipeline | None" = None
_FLUX_PIPELINE_LOCK = asyncio.Lock()
_ACTIVE_JOBS: dict[str, dict[str, object]] = {}


def _job_key(channel: str, chat_id: str) -> str:
    return f"{channel}:{chat_id}"


def is_image_job_active(channel: str, chat_id: str) -> bool:
    job = _ACTIVE_JOBS.get(_job_key(channel, chat_id))
    if not job:
        return False
    task = job.get("task")
    return isinstance(task, asyncio.Task) and not task.done()


def get_image_job_status(channel: str, chat_id: str) -> str | None:
    job = _ACTIVE_JOBS.get(_job_key(channel, chat_id))
    if not job:
        return None
    started_at = job.get("started_at")
    tool = job.get("tool")
    prompt = job.get("prompt")
    return f"Image generation in progress ({tool}). Started at {started_at}. Prompt: {prompt}"


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


def _select_flux_dtype() -> torch.dtype:
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _load_flux_pipeline(model_id: str) -> "Flux2KleinPipeline":
    if Flux2KleinPipeline is None:
        raise RuntimeError(
            "Flux2KleinPipeline not available. Upgrade diffusers to a version that includes it."
        )
    dtype = _select_flux_dtype()
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe


async def _get_flux_pipeline(model_id: str) -> "Flux2KleinPipeline":
    global _FLUX_PIPELINE
    if _FLUX_PIPELINE is not None:
        return _FLUX_PIPELINE

    async with _FLUX_PIPELINE_LOCK:
        if _FLUX_PIPELINE is not None:
            return _FLUX_PIPELINE
        loop = asyncio.get_running_loop()
        _FLUX_PIPELINE = await loop.run_in_executor(None, _load_flux_pipeline, model_id)
        return _FLUX_PIPELINE


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
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Optional negative prompt to avoid unwanted qualities",
                },
            },
            "required": ["prompt"],
        }

    def _generate_image(
        self,
        pipe: StableDiffusionXLPipeline,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
    ) -> Path:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt
            or "low quality, worst quality, watermark",
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
            width=512,
            height=512,
        )
        image = result.images[0]

        media_dir = Path.home() / ".nanobot" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        filename = f"sdxl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = media_dir / filename
        image.save(file_path)
        return file_path

    async def _run_job(
        self,
        channel: str,
        chat_id: str,
        prompt: str,
        negative_prompt: str | None,
        seed: int,
    ) -> None:
        try:
            pipe = await _get_pipeline(self._model_id)
            file_path = await asyncio.to_thread(
                self._generate_image, pipe, prompt, negative_prompt, seed
            )
            msg = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=f"Here is your SDXL image. (seed={seed})",
                media=[str(file_path)],
            )
            await self._send_callback(msg)
        except Exception as e:
            logger.error(f"SDXL generation failed: {e}")
            if self._send_callback:
                await self._send_callback(
                    OutboundMessage(
                        channel=channel,
                        chat_id=chat_id,
                        content=f"Error: SDXL generation failed: {e}",
                    )
                )
        finally:
            _ACTIVE_JOBS.pop(_job_key(channel, chat_id), None)

    async def execute(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        **kwargs: object,
    ) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        if not self._send_callback:
            return "Error: Message sending not configured"

        if is_image_job_active(channel, chat_id):
            return "Image generation is already in progress. Reply 'continue' for status."

        seed = random.randint(0, 2**31 - 1)
        task = asyncio.create_task(
            self._run_job(channel, chat_id, prompt, negative_prompt, seed)
        )
        _ACTIVE_JOBS[_job_key(channel, chat_id)] = {
            "task": task,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "tool": "sdxl",
            "prompt": prompt[:200],
        }
        return "Started SDXL image generation. I'll send the result when it's ready."


class Flux2KleinBase9BTool(Tool):
    """Generate or edit an image using FLUX.2 [klein] 9B Base."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._model_id = os.environ.get(
            "FLUX2_KLEIN_BASE_9B_MODEL_ID",
            "black-forest-labs/FLUX.2-klein-base-9B",
        )

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "generate_image_flux2_klein_base_9b"

    @property
    def description(self) -> str:
        return "Generate an image using FLUX.2 [klein] 9B Base, optionally with a reference image."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the desired image",
                    "minLength": 1,
                },
                "reference_image": {
                    "type": "string",
                    "description": "Optional local file path to a reference image",
                },
            },
            "required": ["prompt"],
        }

    def _generate_image(
        self,
        pipe: "Flux2KleinPipeline",
        prompt: str,
        reference_image: list[Image.Image] | None,
        seed: int,
    ) -> Path:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        args: dict[str, object] = {
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "guidance_scale": 4.0,
            "num_inference_steps": 50,
            "generator": generator,
        }
        if reference_image is not None:
            args["image"] = reference_image
        result = pipe(**args)
        image = result.images[0]

        media_dir = Path.home() / ".nanobot" / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        filename = f"flux2_klein_base_9b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = media_dir / filename
        image.save(file_path)
        return file_path

    async def _run_job(
        self,
        channel: str,
        chat_id: str,
        prompt: str,
        reference_image: list[Image.Image] | None,
        seed: int,
    ) -> None:
        try:
            pipe = await _get_flux_pipeline(self._model_id)
            file_path = await asyncio.to_thread(
                self._generate_image, pipe, prompt, reference_image, seed
            )
            msg = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=f"Here is your FLUX.2 [klein] 9B Base image. (seed={seed})",
                media=[str(file_path)],
            )
            await self._send_callback(msg)
        except Exception as e:
            logger.error(f"FLUX.2 generation failed: {e}")
            if self._send_callback:
                await self._send_callback(
                    OutboundMessage(
                        channel=channel,
                        chat_id=chat_id,
                        content=f"Error: FLUX.2 generation failed: {e}",
                    )
                )
        finally:
            _ACTIVE_JOBS.pop(_job_key(channel, chat_id), None)

    async def execute(
        self,
        prompt: str,
        reference_image: str | None = None,
        **kwargs: object,
    ) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        if not self._send_callback:
            return "Error: Message sending not configured"

        if is_image_job_active(channel, chat_id):
            return "Image generation is already in progress. Reply 'continue' for status."

        seed = random.randint(0, 2**31 - 1)

        image_input: list[Image.Image] | None = None
        if reference_image:
            ref_path = Path(reference_image)
            if not ref_path.exists():
                return f"Error: reference_image not found: {reference_image}"
            image_input = [Image.open(ref_path).convert("RGB")]

        task = asyncio.create_task(
            self._run_job(channel, chat_id, prompt, image_input, seed)
        )
        _ACTIVE_JOBS[_job_key(channel, chat_id)] = {
            "task": task,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "tool": "flux2-klein-base-9b",
            "prompt": prompt[:200],
        }
        return "Started FLUX.2 image generation. I'll send the result when it's ready."
