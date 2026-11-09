from typing import Literal, TypedDict


Role = Literal["system", "user", "assistant"]
class GradioChatHistory(TypedDict):
    role: Role
    content: str
