from typing import Tuple
import inspect
from langchain.llms.base import LLM
from openai import AsyncOpenAI
from llm.llm_provider import LLMProvider, QueueCallback
from queue import Queue
import os
import httpx

class OpenShiftAIvLLM(LLMProvider):
  def __init__(self, provider, model, params):
    super().__init__(provider, model, params)
    pass

  def _openshift_ai_vllm_instance(self) -> LLM:
    print(f"[{inspect.stack()[0][3]}] Creating OpenShift AI vLLM instance")
    try:
      from langchain_community.llms.vllm import VLLMOpenAI
    except Exception as e:
      print(
          "Missing vLLM libraries. VLLMOpenAI provider will be unavailable."
      )
      raise e
    creds = self._get_llm_credentials()
    if creds in (None, ''):
        creds = "dummy-api-key" # ChatOpenAI requires creds to be not none
    if self._llm_instance is None:
      # Creating an object of custom handler
      params: dict = {
          "base_url": self._get_llm_url(""),
          "model": self.model,
          "timeout": None,
          "cache": None,
          "streaming": True,
          "temperature": 0.01,
          "max_tokens": 512,
          #"top_p": 0.95,
          "verbose": True,
          "callbacks": [QueueCallback(self._queue)]
      }
      os.environ["OPENAI_API_KEY"] =  creds
      async_client=httpx.AsyncClient(verify=False)
      http_client=httpx.Client(verify=False)
      self._llm_instance = VLLMOpenAI(**params, async_client=async_client, http_client=http_client)

    print(f"[{inspect.stack()[0][3]}] OpenShift AI vLLM instance {self._llm_instance}")
    return self._llm_instance

  def get_llm(self) -> Tuple[LLM, Queue]:
    return self._openshift_ai_vllm_instance(), self._queue