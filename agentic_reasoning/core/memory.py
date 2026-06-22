# core/memory.py

from typing import Dict, List, Tuple, Optional

class HybridMemory:
    """
     "Thought"  "Observation" 。
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._thought_counter = 0
        self._obs_counter = 0
        self._thoughts: Dict[str, Dict[str, object]] = {}
        self._observations: Dict[str, Dict[str, object]] = {}
        print("[Memory] Hybrid memory (Thought/Observation) has been reset.")

    def _generate_thought_id(self) -> str:
        self._thought_counter += 1
        return str(self._thought_counter)

    def _generate_obs_id(self) -> str:
        self._obs_counter += 1
        return str(self._obs_counter)

    def record_thought(
        self,
        detailed_reasoning: str,
        summary_text: str,
        tool_calls: Optional[list] = None
    ) -> str:
        """
        - detailed_reasoning: （original）
        - summary_text: （summary）
        - tool_calls: ， LLM  tool_calls ， assistant 。
        """
        thought_id = self._generate_thought_id()
        record: Dict[str, object] = {
            "original": detailed_reasoning,
            "summary": summary_text,
        }
        if tool_calls is not None:
            record["tool_calls"] = tool_calls

        self._thoughts[thought_id] = record
        print(f"[Memory] Thought recorded. ID: {thought_id}.")
        return thought_id

    def record_observation(
        self,
        observation_summary: str,
        raw_observation: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[dict] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """
        “”。

        -  (observation_summary, raw_observation) ， summary / original。
        -  tool_name / tool_args / tool_call_id，。
        """
        obs_id = self._generate_obs_id()
        record: Dict[str, object] = {
            "summary": observation_summary,
            "original": str(raw_observation),
        }

        if tool_name is not None:
            record["tool_name"] = tool_name
        if tool_args is not None:
            record["tool_args"] = tool_args
        if tool_call_id is not None:
            record["tool_call_id"] = tool_call_id

        self._observations[obs_id] = record
        print(f"[Memory] Structured Observation recorded. ID: {obs_id}.")
        return obs_id

    def get_thought(self, thought_id: str) -> Dict[str, object]:
        """ID。"""
        return self._thoughts.get(str(thought_id), {})

    def get_observation(self, obs_id: str) -> Dict[str, object]:
        """ID。"""
        return self._observations.get(str(obs_id), {})


    def update_thought_summary(self, thought_id: str, summary_text: str) -> None:
        """ summary 。"""
        record = self._thoughts.get(str(thought_id))
        if record is not None:
            record["summary"] = summary_text
            self._thoughts[str(thought_id)] = record

    def update_observation_summary(self, obs_id: str, summary_text: str) -> None:
        """ summary 。"""
        record = self._observations.get(str(obs_id))
        if record is not None:
            record["summary"] = summary_text
            self._observations[str(obs_id)] = record


    def end_step(
        self,
        detailed_reasoning: str,
        summary_text: str,
        reasoning_content_from_msg: Optional[str] = None
    ) -> str:
        print("[Memory] (Compatibility Mode) end_step called.")
        thought_id = self.record_thought(detailed_reasoning, summary_text)

        if reasoning_content_from_msg:
            self._thoughts[thought_id]['metadata'] = {
                "reasoning_content_from_msg": reasoning_content_from_msg
            }

        return thought_id

    def get_summary_data(self, summary_id: str) -> Dict[str, object]:
        print(f"[Memory] (Compatibility Mode) get_summary_data called for ID: {summary_id}.")
        return self.get_thought(summary_id)


    def get_all_intents(self) -> List[Tuple[str, Dict[str, object]]]:
        return list(self._thoughts.items())

    def get_all_observations(self) -> List[Tuple[str, Dict[str, object]]]:
        return list(self._observations.items())

    def delete_thought(self, thought_id: str):
        """ thought （）"""
        if str(thought_id) in self._thoughts:
            del self._thoughts[str(thought_id)]
            print(f"[Memory] Thought {thought_id} deleted.")