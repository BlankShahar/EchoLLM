from .config import LatencyConfig, LatencyMode


class LatencyModel:
    def __init__(self, config: LatencyConfig):
        self.config = config

    def estimate_ms(self, prompt: str, response: str) -> float:
        if self.config.mode == LatencyMode.FIXED:
            return self.config.fixed_ms
        prompt_tokens = max(1, len(prompt.split()))
        response_tokens = max(1, len(response.split()))
        return (
            self.config.base_ms
            + prompt_tokens * self.config.prompt_token_ms
            + response_tokens * self.config.response_token_ms
        )
