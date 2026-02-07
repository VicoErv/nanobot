"""Audio generation tools."""

import asyncio
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.media_jobs import clear_job, is_job_active, register_job
from nanobot.bus.events import OutboundMessage

try:
    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
except Exception:  # pragma: no cover - optional dependency version
    AceStepHandler = None  # type: ignore[assignment]
    LLMHandler = None  # type: ignore[assignment]
    GenerationConfig = None  # type: ignore[assignment]
    GenerationParams = None  # type: ignore[assignment]
    generate_music = None  # type: ignore[assignment]


_ACE_HANDLER: "AceStepHandler | None" = None
_ACE_LLM_HANDLER: "LLMHandler | None" = None
_ACE_LOCK = asyncio.Lock()


async def _get_ace_handlers(model_name: str) -> tuple["AceStepHandler", "LLMHandler | None"]:
    global _ACE_HANDLER, _ACE_LLM_HANDLER
    if _ACE_HANDLER is not None:
        return _ACE_HANDLER, _ACE_LLM_HANDLER

    async with _ACE_LOCK:
        if _ACE_HANDLER is not None:
            return _ACE_HANDLER, _ACE_LLM_HANDLER

        if AceStepHandler is None or GenerationParams is None:
            raise RuntimeError(
                "AceStep dependencies not available. Install ace-step and its dependencies."
            )

        persistent_storage = Path.home() / ".cache" / "ace-step"
        persistent_storage.mkdir(parents=True, exist_ok=True)

        try:
            handler = AceStepHandler(persistent_storage_path=str(persistent_storage))
        except TypeError:
            handler = AceStepHandler()
        init_kwargs = {
            "project_root": str(persistent_storage),
            "config_path": model_name,
            "device": "auto",
            "use_flash_attention": False,
            "compile_model": False,
            "offload_to_cpu": False,
            "offload_dit_to_cpu": False,
            "quantization": None,
            "shared_vae": False,
            "shared_text_encoder": False,
            "shared_text_tokenizer": False,
            "shared_silence_latent": False,
        }
        try:
            import inspect

            sig = inspect.signature(handler.initialize_service)
            allowed = set(sig.parameters.keys())
            init_kwargs = {k: v for k, v in init_kwargs.items() if k in allowed}
        except Exception:
            pass

        handler.initialize_service(**init_kwargs)

        llm_handler: LLMHandler | None = None
        if LLMHandler is not None and os.environ.get("ACESTEP_USE_LM", "1") == "1":
            try:
                try:
                    llm_handler = LLMHandler(persistent_storage_path=str(persistent_storage))
                except TypeError:
                    llm_handler = LLMHandler()
                checkpoint_dir = Path(persistent_storage) / "checkpoints"
                llm_handler.initialize(
                    checkpoint_dir=str(checkpoint_dir),
                    model_name=os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-1.7B"),
                    backend="pt",
                    device="auto",
                )
            except Exception as e:
                logger.warning(f"AceStep LLM init failed, continuing without LLM: {e}")
                llm_handler = None

        _ACE_HANDLER = handler
        _ACE_LLM_HANDLER = llm_handler
        return handler, llm_handler


class AceStepTurboTextToAudioTool(Tool):
    """Generate audio using ACE-Step 1.5 Turbo (text-to-audio)."""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id
        self._model_name = os.environ.get("ACESTEP_MODEL", "acestep-v15-turbo")
        self._audio_format = os.environ.get("ACESTEP_AUDIO_FORMAT", "flac")

    def set_context(self, channel: str, chat_id: str) -> None:
        self._default_channel = channel
        self._default_chat_id = chat_id

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        self._send_callback = callback

    @property
    def name(self) -> str:
        return "generate_audio_acestep_v15_turbo"

    @property
    def description(self) -> str:
        return "Generate audio using ACE-Step 1.5 Turbo (text-to-audio)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt describing the desired audio",
                    "minLength": 1,
                },
                "lyrics": {
                    "type": "string",
                    "description": "Optional lyrics (use [Instrumental] for no vocals)",
                },
                "duration": {
                    "type": "number",
                    "description": "Optional duration in seconds",
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["prompt"],
        }

    def _generate_audio(
        self,
        handler: "AceStepHandler",
        llm_handler: "LLMHandler | None",
        prompt: str,
        lyrics: str | None,
        duration: float | None,
        seed: int,
        output_dir: Path,
    ) -> Path:
        if GenerationParams is None or GenerationConfig is None or generate_music is None:
            raise RuntimeError("AceStep generation API not available.")

        params_kwargs = {
            "caption": prompt,
            "lyrics": lyrics or "[Instrumental]",
            "use_audio_prompt": False,
            "audio_prompt_path": "",
            "duration": duration,
            "seed": seed,
            "inference_steps": 8,
        }
        try:
            import inspect

            sig = inspect.signature(GenerationParams)
            allowed = set(sig.parameters.keys())
            params_kwargs = {k: v for k, v in params_kwargs.items() if k in allowed}
        except Exception:
            pass

        params = GenerationParams(**params_kwargs)

        config_kwargs = {
            "output_dir": str(output_dir),
            "output_name": f"acestep_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "sample_rate": 44100,
            "audio_format": self._audio_format,
            "use_random_seed": True,
        }
        try:
            import inspect

            sig = inspect.signature(GenerationConfig)
            allowed = set(sig.parameters.keys())
            config_kwargs = {k: v for k, v in config_kwargs.items() if k in allowed}
        except Exception:
            pass

        config = GenerationConfig(**config_kwargs)

        results = generate_music(
            params=params,
            config=config,
            handler=handler,
            llm_handler=llm_handler,
            opt_handler=None,
            load_audio=False,
            use_tqdm=False,
        )

        if not results or not results[0].audio_path:
            raise RuntimeError("ACE-Step returned no audio.")

        return Path(results[0].audio_path)

    async def _run_job(
        self,
        channel: str,
        chat_id: str,
        prompt: str,
        lyrics: str | None,
        duration: float | None,
        seed: int,
    ) -> None:
        try:
            handler, llm_handler = await _get_ace_handlers(self._model_name)
            output_dir = Path.home() / ".nanobot" / "media"
            output_dir.mkdir(parents=True, exist_ok=True)
            audio_path = await asyncio.to_thread(
                self._generate_audio,
                handler,
                llm_handler,
                prompt,
                lyrics,
                duration,
                seed,
                output_dir,
            )

            msg = OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=f"Here is your ACE-Step audio. (seed={seed})",
                media=[str(audio_path)],
            )
            await self._send_callback(msg)
        except Exception as e:
            logger.error(f"ACE-Step generation failed: {e}")
            if self._send_callback:
                await self._send_callback(
                    OutboundMessage(
                        channel=channel,
                        chat_id=chat_id,
                        content=f"Error: ACE-Step generation failed: {e}",
                    )
                )
        finally:
            clear_job(channel, chat_id)

    async def execute(
        self,
        prompt: str,
        lyrics: str | None = None,
        duration: float | None = None,
        **kwargs: object,
    ) -> str:
        channel = self._default_channel
        chat_id = self._default_chat_id

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        if not self._send_callback:
            return "Error: Message sending not configured"

        if is_job_active(channel, chat_id):
            await self._send_callback(
                OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content="Audio generation is already in progress. Reply 'continue' for status.",
                )
            )
            return "already_running"

        seed = random.randint(0, 2**31 - 1)
        task = asyncio.create_task(
            self._run_job(channel, chat_id, prompt, lyrics, duration, seed)
        )
        register_job(
            channel=channel,
            chat_id=chat_id,
            task=task,
            tool="acestep-v15-turbo",
            prompt=prompt,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

        await self._send_callback(
            OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content="Started ACE-Step audio generation. I'll send the result when it's ready.",
            )
        )
        return "started"
