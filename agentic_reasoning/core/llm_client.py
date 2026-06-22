import json, logging, time
from typing import Dict, List, Any, Union, Iterator
import sys
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage, ChatCompletionChunk
import logging
import os


class LLMClient:
    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.example.com/v1",
        api_port: int = 8000,
        *,
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 32768,
        max_retries: int = 10,
        retry_delay: float = 3.0,
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        self.base_url = base_url
        self.api_key = api_key
        # self.max_completion_tokens = max_tokens

        if not self.base_url:
            self.base_url = os.environ.get("REASONING_ENDPOINT", "")
        if not self.api_key:
            self.api_key = os.environ.get("REASONING_SK", "")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        
       
    def call_llm(
            self,
            messages: List[Dict],
            *,
            tools: List[Dict] = None,
            extra_body: Dict = None,
            tool_choice: Union[str, Dict] = None,
        ) -> Union[ChatCompletionMessage, Iterator[ChatCompletionChunk]]:
            """
             LLM API ，，。

            Args:
                messages (List[Dict]): 。
                tools (List[Dict], optional): 。Defaults to None.
                extra_body (Dict, optional):  API 。Defaults to None.

            """
            params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": 1.1,
            }
            if tools:
                logging.info(f"[LLMClient] Using {len(tools)} tools in the request.")
                params["tools"] = tools     
            if tool_choice:
                logging.info(f"[LLMClient] Setting tool_choice to: {tool_choice}")
                params["tool_choice"] = tool_choice

            eb = {}

            if extra_body:
                eb.update(extra_body)

            if eb:
                logging.info(f"[LLMClient] Adding extra body parameters: {eb}")
                params["extra_body"] = eb
            


            for attempt in range(self.max_retries):
                try: 
                    response = self.client.chat.completions.create(**params)

                    if response.choices and response.choices[0].message.role == "assistant":
                        if response.choices[0].message.content or response.choices[0].message.tool_calls:
                            logging.info("Output generated sucessfully!")
                            # print("Output generated sucessfully!")
                            return response.choices[0].message 
                        else:
                            # print("Response content is empty, retrying...")
                            # print(response)
                            logging.warning(f"Attempt {attempt + 1}: Content is empty, retrying...")

                except Exception as e:
                    logging.warning(f"[LLMClient] API call attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        logging.error("[LLMClient] All API call retries failed.")

   
