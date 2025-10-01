import os
import json
import logging
from typing import Optional, Tuple, Type, Union
from pathlib import Path

import numpy as np
from omgenerate import OMGeneration, CausalLMOutput
from PyRuntime import OMExecutionSession

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
DLC_DECODER_NAME = "decoder_model.so"
DLC_DECODER_WITH_PAST_NAME = "decoder_with_past_model.so"
MULTI_QUERY_ATTN_MODELS = {"gpt_bigcode"}


class OMConfig:
    """Simple config class to replace transformers config dependencies."""

    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> "OMConfig":
        """Load config from JSON file."""
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class OMGenerationConfig:
    """Simple generation config to replace transformers GenerationConfig."""

    def __init__(self, **kwargs):
        # Default values
        self.max_length = kwargs.get("max_length", 50)
        self.max_new_tokens = kwargs.get("max_new_tokens", None)
        self.min_length = kwargs.get("min_length", 0)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.eos_token_id = kwargs.get("eos_token_id", None)

        # Set any additional kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    @classmethod
    def from_model_config(cls, config: OMConfig) -> "OMGenerationConfig":
        """Create generation config from model config."""
        config_dict = (
            config.to_dict() if hasattr(config, "to_dict") else config.__dict__
        )
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, config_path: Union[str, Path]) -> "OMGenerationConfig":
        """Load generation config from JSON file."""
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            logger.info("Generation config file not found, using defaults.")
            return cls()


class OMPreTrainedModel:
    config_name = CONFIG_NAME

    def __init__(self, config: OMConfig):
        super().__init__()
        self.config = config

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward pass of the model, needs to be overwritten."""
        raise NotImplementedError

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: OMConfig,
        **kwargs,
    ) -> "OMPreTrainedModel":
        raise NotImplementedError(
            "Overwrite this method in subclass to define how to load your model from pretrained"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Optional[OMConfig] = None,
        **kwargs,
    ) -> "OMPreTrainedModel":
        """Load model from pretrained weights."""
        model_path = Path(model_id)

        if config is None:
            config_path = model_path / CONFIG_NAME
            if config_path.exists():
                config = OMConfig.from_json_file(config_path)
            else:
                raise OSError(f"config.json not found in {model_id}")
        elif isinstance(config, (str, os.PathLike)):
            config = OMConfig.from_json_file(config)

        return cls._from_pretrained(model_id=model_id, config=config, **kwargs)

    @staticmethod
    def load_model(path: Union[str, Path], tag: str) -> OMExecutionSession:
        if not isinstance(path, str):
            path = str(path)

        return OMExecutionSession(path, tag=tag)


class OMInferSession:
    """Session to invoke one forward pass at a time using DLC."""

    def __init__(self, session: OMExecutionSession, parent_model: OMPreTrainedModel):
        self.session = session
        self.parent_model = parent_model
        self.input_names = [
            x["name"] for x in json.loads(self.session.input_signature())
        ]
        self.output_names = {
            x["name"]: idx
            for idx, x in enumerate(json.loads(self.session.output_signature()))
        }

    def forward(self, *args, **kwargs):
        """Forward pass - should be implemented by subclasses."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OMDecoder(OMInferSession):
    """Decoder model with language modeling head for ONNX-MLIR inference."""

    def __init__(self, session: OMExecutionSession, parent_model: OMPreTrainedModel):
        super().__init__(session, parent_model)
        self.key_value_output_names = [
            key for key in self.output_names if (".key" in key) or (".value" in key)
        ]
        self.key_value_input_names = [
            key for key in self.input_names if (".key" in key) or (".value" in key)
        ]

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[Tuple[np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
    ) -> CausalLMOutput:

        input_ids = np.ascontiguousarray(input_ids, dtype=np.int64)
        if attention_mask is not None:
            attention_mask = np.ascontiguousarray(attention_mask, dtype=np.int64)
        if labels is not None:
            labels = np.ascontiguousarray(labels, dtype=np.int64)

        # Flatten the past_key_values (no need to flatten for models using multi-query attn)
        if (
            past_key_values is not None
            and getattr(self.parent_model.config, "model_type", "")
            not in MULTI_QUERY_ATTN_MODELS
        ):
            past_key_values = [
                pkv for pkv_per_layer in past_key_values for pkv in pkv_per_layer
            ]

        # prepare inputs in correct order
        inputs = []
        pkv_idx = 0
        for name in self.input_names:
            if name == "input_ids":
                inputs.append(input_ids)
            elif name == "attention_mask":
                inputs.append(attention_mask)
            elif name == "labels":
                inputs.append(labels)
            elif "past_key_values" in name and past_key_values is not None:
                inputs.append(past_key_values[pkv_idx])
                pkv_idx += 1

        # run inference session
        outputs = self.session.run(inputs)

        # Process outputs - they're already numpy arrays from ONNX-MLIR
        past_key_values = [
            outputs[self.output_names[key]] for key in self.key_value_output_names
        ]

        # Restructure past_key_values for non-multi-query models
        if (
            getattr(self.parent_model.config, "model_type", "")
            not in MULTI_QUERY_ATTN_MODELS
        ):
            past_key_values = [
                past_key_values[i : i + 2] for i in range(0, len(past_key_values), 2)
            ]

        logits = outputs[self.output_names["logits"]]

        return CausalLMOutput(logits=logits, past_key_values=past_key_values)


class OMModelDecoder(OMPreTrainedModel):
    """Base class for implementing models with a causal language modeling head using ONNX-MLIR inference."""

    def __init__(
        self,
        decoder_session: OMExecutionSession,
        config: OMConfig,
        decoder_with_past_session: Optional[OMExecutionSession] = None,
        use_cache: bool = True,
        generation_config: Optional[OMGenerationConfig] = None,
        **kwargs,
    ):
        # Explicitly call OMPreTrainedModel.__init__ to ensure config is set
        OMPreTrainedModel.__init__(self, config)

        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        # Validate cache configuration
        if use_cache and decoder_with_past_session is None:
            raise ValueError("use_cache=True requires decoder_with_past_session")
        if not use_cache and decoder_with_past_session is not None:
            raise ValueError("decoder_with_past_session provided but use_cache=False")

        self.use_cache = use_cache
        self.decoder = OMDecoder(decoder_session, self)
        self.decoder_with_past = (
            OMDecoder(decoder_with_past_session, self) if use_cache else None
        )

        self.generation_config = (
            generation_config or OMGenerationConfig.from_model_config(config)
        )

    @staticmethod
    def load_model(
        decoder_path: Union[str, Path],
        decoder_with_past_path: Optional[Union[str, Path]] = None,
    ):
        """
        Creates ONNX-MLIR inference sessions.
        Two inference sessions will be created for respectively the decoder and decoder with past key values
        models.

        Args:
            decoder_path (`str` or `Path`):
                The path of the decoder ONNX-MLIR shared library..
            decoder_with_past_path (`str` or `Path`, *optional*):
                The path of the decoder with past key values ONNX-MLIR shared library..
        """

        decoder_session = OMPreTrainedModel.load_model(decoder_path, tag="decoder")
        decoder_with_past_session = (
            OMPreTrainedModel.load_model(
                decoder_with_past_path, tag="decoder_with_past"
            )
            if decoder_with_past_path
            else None
        )

        return decoder_session, decoder_with_past_session

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: OMConfig,
        init_cls,
        decoder_file_name: str = DLC_DECODER_NAME,
        decoder_with_past_file_name: str = DLC_DECODER_WITH_PAST_NAME,
        use_cache: bool = True,
        **kwargs,
    ):
        # Load model from pretrained directory
        model_path = Path(model_id)

        # Build base paths (without extensions)
        decoder_path = model_path / decoder_file_name
        decoder_with_past_path = (
            model_path / decoder_with_past_file_name if use_cache else None
        )

        # Build .so file paths
        decoder_so = decoder_path.with_suffix(".so")
        decoder_with_past_so = (
            decoder_with_past_path.with_suffix(".so") if use_cache else None
        )

        # Validate .so files exist
        if not decoder_so.exists():
            raise ValueError("Not found the .so file for the decoder model.")
        if use_cache and not decoder_with_past_so.exists():
            raise ValueError("Not found the .so file for the decoder_with_past model.")

        # Load sessions
        decoder_session, decoder_with_past_session = cls.load_model(
            decoder_path=decoder_so,
            decoder_with_past_path=decoder_with_past_so,
        )

        generation_config = OMGenerationConfig.from_json_file(
            model_path / "generation_config.json"
        )

        return init_cls(
            decoder_session,
            config,
            decoder_with_past_session=decoder_with_past_session,
            use_cache=use_cache,
            generation_config=generation_config,
        )


class OMModelForCausalLM(OMModelDecoder, OMGeneration):
    """ONNX model with a causal language modeling head for ONNX-MLIR inference."""

    def forward(
        self,
        input_ids: np.ndarray = None,
        attention_mask: Optional[np.ndarray] = None,
        past_key_values: Optional[Tuple[np.ndarray]] = None,
        labels: Optional[np.ndarray] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutput:
        # Use instance use_cache if not specified
        use_cache = use_cache if use_cache is not None else self.use_cache

        if past_key_values is None or not use_cache:
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=labels,
            )
        else:
            outputs = self.decoder_with_past(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                labels=labels,
            )

        return CausalLMOutput(
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: OMConfig,
        **kwargs,
    ):
        return super()._from_pretrained(
            model_id, config, init_cls=OMModelForCausalLM, **kwargs
        )
