from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    offline_fallback: bool = False


class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    hostname: Optional[str] = None
    os_name: Optional[str] = None


class EnrollResponse(BaseModel):
    device_id: str
    agent_token: str
    name: str


class HeartbeatRequest(BaseModel):
    inventory: dict[str, Any] = Field(default_factory=dict)
    status: str = "online"


class ActionCreateRequest(BaseModel):
    device_id: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class ActionResultRequest(BaseModel):
    status: Literal["succeeded", "failed", "rejected"]
    output: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class HardenRequest(BaseModel):
    topic: str
    context: str = ""
    language: Literal["fa", "en"] = "en"


class VendorSyncRequest(BaseModel):
    vendor_ids: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)