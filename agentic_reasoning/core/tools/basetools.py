# core/tools/base.py

from abc import ABC, abstractmethod
from typing import Optional, Union

class BaseTool(ABC):
    """"""
    name: str = ""
    description: str = ""
    parameters: dict = {}

    def __init__(self, cfg: Optional[dict] = None):  
        self.cfg = cfg or {}

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> str:
        raise NotImplementedError

    def _get_schema(self) -> dict:
        """OpenAI schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
