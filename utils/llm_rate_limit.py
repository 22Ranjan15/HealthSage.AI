import asyncio
import time
from typing import Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatResult

class RateLimitedGemini(ChatGoogleGenerativeAI):
    delay_seconds: float = 10.0  # 10 seconds = 6 requests per minute

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Sleep before making the request
        time.sleep(self.delay_seconds)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Async Sleep before making the request
        await asyncio.sleep(self.delay_seconds)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)