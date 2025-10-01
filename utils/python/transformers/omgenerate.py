import numpy as np
from typing import Optional, Union, List


class OMGeneration:
    """
    Pure NumPy generation mixin.
    Provides basic text generation capabilities for autoregressive models.
    """

    def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        max_length: int = 50,
        max_new_tokens: Optional[int] = None,
        min_length: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        pad_token_id: int = 0,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        use_cache: bool = True,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate sequences using the model using only numpy.

        Args:
            input_ids: Input token ids of shape [batch_size, sequence_length]
            attention_mask: Attention mask of shape [batch_size, sequence_length]
            max_length: Maximum length of generated sequence (including input sequence)
            max_new_tokens: Maximum number of new tokens to generate
            min_length: Minimum length of generated sequence
            do_sample: Whether to use sampling instead of greedy decoding
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            pad_token_id: Token ID for padding
            eos_token_id: Token ID(s) for end of sequence
            use_cache: Whether to use key-value caching
            random_seed: Random seed for reproducible sampling

        Returns:
            Generated token sequences of shape [batch_size, generated_length]
        """
        # random seed for reproducibility
        if random_seed is not None:
            rng = np.random.Generator(np.random.PCG64(random_seed))
        else:
            rng = np.random.default_rng()

        # Validate parameters at the start
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be between 0 and 1")

        if top_k is not None and top_p is not None:
            raise ValueError("Cannot use both top_k and top_p simultaneously")

        # Validate input shapes
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape must match input_ids shape")

        input_ids = np.asarray(input_ids, dtype=np.int64)
        if attention_mask is not None:
            attention_mask = np.asarray(attention_mask, dtype=np.int64)

        # Get generation config defaults
        generation_config = getattr(self, "generation_config", None)
        if generation_config is not None:
            max_length = max_length or getattr(generation_config, "max_length", 50)
            max_new_tokens = max_new_tokens or getattr(
                generation_config, "max_new_tokens", None
            )
            min_length = min_length or getattr(generation_config, "min_length", 0)
            pad_token_id = pad_token_id or getattr(generation_config, "pad_token_id", 0)
            eos_token_id = eos_token_id or getattr(
                generation_config, "eos_token_id", None
            )

        # Set max_length based on max_new_tokens if provided
        if max_new_tokens is not None:
            max_length = input_ids.shape[-1] + max_new_tokens

        # Ensure eos_token_id is a list
        if eos_token_id is not None and not isinstance(eos_token_id, (list, tuple)):
            eos_token_id = [eos_token_id]

        batch_size, cur_len = input_ids.shape

        # Initialize variables for generation
        generated_ids = input_ids.copy()
        past_key_values = None
        finished_sequences = np.zeros(batch_size, dtype=bool)
        attention_mask = (
            attention_mask
            if attention_mask is not None
            else np.ones_like(input_ids, dtype=np.int64)
        )

        # Generation loop
        while cur_len < max_length:
            if past_key_values is None or not use_cache:
                forward_input_ids = generated_ids
            else:
                forward_input_ids = generated_ids[:, -1:]

            # Forward pass
            outputs = self.forward(
                input_ids=forward_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values if use_cache else None,
            )

            # Get next token logits and update cache
            next_token_logits = outputs.logits[:, -1, :]
            if use_cache:
                past_key_values = outputs.past_key_values

            # Generate next tokens
            if do_sample:
                next_tokens = self._sample_next_tokens(
                    next_token_logits, temperature, top_k, top_p, rng=rng
                )
            else:
                next_tokens = np.argmax(next_token_logits, axis=-1)

            # Don't update finished sequences
            next_tokens = np.where(finished_sequences, pad_token_id, next_tokens)

            # Append next tokens
            generated_ids = np.concatenate(
                [generated_ids, next_tokens.reshape(-1, 1)], axis=1
            )

            # Check for finished sequences (EOS token or already finished)
            if eos_token_id is not None:
                for eos_id in eos_token_id:
                    finished_sequences |= next_tokens == eos_id

            # Update attention mask
            attention_mask = np.concatenate(
                [attention_mask, (~finished_sequences).astype(np.int64).reshape(-1, 1)],
                axis=1,
            )

            cur_len += 1

            # Check if all sequences are finished or minimum length reached
            if cur_len >= min_length and finished_sequences.all():
                break

        return generated_ids

    def _compute_probs(self, logits: np.ndarray) -> np.ndarray:
        """Compute probabilities with numerical stability."""
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _sample_next_tokens(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        rng=None,
    ) -> np.ndarray:
        batch_size, vocab_size = logits.shape
        rng = rng or np.random.default_rng()

        if temperature == 0.0:
            # Temperature = 0 means division by 0 -> mathematically not defined
            # so we handle it as greedy decoding
            return np.argmax(logits, axis=-1)
        elif temperature != 1.0:
            logits = logits / temperature

        # Apply filtering
        if top_k is not None and top_k > 0:
            logits = self._top_k_filtering(logits, top_k)
        elif top_p is not None and 0.0 < top_p < 1.0:
            logits = self._top_p_filtering(logits, top_p)

        probs = self._compute_probs(logits)

        # Sample from distribution
        next_tokens = np.array(
            [rng.choice(vocab_size, p=probs[i]) for i in range(batch_size)]
        )

        return next_tokens

    def _top_k_filtering(self, logits: np.ndarray, top_k: int) -> np.ndarray:
        """Keep only top-k tokens, set others to -inf."""
        top_k_values = np.partition(logits, -top_k, axis=-1)[:, -top_k:]
        min_values = np.min(top_k_values, axis=-1, keepdims=True)

        # Set values below the k-th largest to -inf
        logits = np.where(logits >= min_values, logits, -np.inf)
        return logits

    def _top_p_filtering(self, logits: np.ndarray, top_p: float) -> np.ndarray:
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            # Sort in descending order
            sorted_indices = np.argsort(logits[i])[::-1]
            sorted_logits = logits[i][sorted_indices]
            sorted_probs = self._compute_probs(sorted_logits.reshape(1, -1))[0]

            # Find cutoff point while keeping at least one token
            cumulative_probs = np.cumsum(sorted_probs)
            sorted_indices_to_remove = cumulative_probs > top_p
            if len(sorted_indices_to_remove) > 1:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False

            # Map back to original indices and set to -inf
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[i][indices_to_remove] = -np.inf

        return logits

    def can_generate(self) -> bool:
        return True


class CausalLMOutput:
    """Simplified output class for numpy arrays."""

    def __init__(self, logits=None, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values  # for caching

    def get(self, key, default=None):
        return getattr(self, key, default)
