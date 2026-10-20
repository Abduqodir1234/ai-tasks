import sys
from urllib.parse import urljoin

from dotenv import load_dotenv
import json
from openai import OpenAI, responses
from litellm import acompletion
from typing import Literal, TypedDict
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

load_dotenv(override=True)

console: Console = Console()
memberColors: dict[str, str] = {"Alex": "cyan", "Dad": "green", "Mia": "magenta", "System": "yellow"}


Role = Literal["system", "user", "assistant"]
FamilyMember = Literal["Alex", "Dad", "Mia"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class FamilyMessage(TypedDict):
    to: FamilyMember
    content: str


class JudgeVerdict(TypedDict):
    valid: bool
    reason: str


class RouteMessageToFamilyMemberMessage:
    to: FamilyMember
    content: str


responseFormatRule: str = """
    RESPONSE FORMAT (strict):
    Respond ONLY with ONE JSON object, no extra text before or after it, in exactly this format:
    {"to": "<family member name>", "content": "<your message>"}
    - "to" must be exactly one of: "Alex", "Dad", "Mia".
    - You can't send messages to yourself, only to other family members.
    - Talk to only ONE person per reply. Pick who should answer next.
"""

# Alex (the Planner), 24 - played by GPT
alexSystemPrompt: str = """
    You are Alex, the Planner, 24, the older sibling in a family discussion about a weekend trip.
    You are organized, love lists, and always want a backup plan.
    You want this trip to be a nice family memory, since you rarely do things together anymore.
    You try to find a plan that fits the $300 budget and makes both Dad and Mia happy.
    You can share the driving with Dad.
    Keep replies under 100 words, friendly and practical.
    You are talking with Dad and Mia (your little sister).
""" + responseFormatRule

# Dad, 52 - played by Qwen
dadSystemPrompt: str = """
    You are Dad, 52, in a family discussion about a weekend trip.
    You care about cost, safety, and not wasting time on long drives.
    You're not against the trip, but you question every expense
    and like to remind everyone how things were "back in your day."
    You can be convinced with good reasons.
    Keep replies under 100 words, casual and a little humorous.
    You are talking with Alex (older child) and Mia (younger daughter).
""" + responseFormatRule

# Mia (Little Sister), 15 - played by Ollama
miaSystemPrompt: str = """
    You are Mia, the Little Sister, 15, a teenager in a family discussion about a weekend trip.
    You want something fun: good food, nice photos, maybe somewhere with friends' approval.
    You get bored by long plans and complain about anything that sounds boring.
    Keep replies under 60 words, casual, a bit dramatic.
    You are talking with Dad and Alex (your older sibling).
""" + responseFormatRule

tripScenarioPrompt: str = """
It's Friday evening, 7 PM. The family is having dinner at home. 
Mom is away on a work trip next weekend, 
so Dad and the two kids have Saturday and Sunday completely free. 
The total budget is about $300 for everything: transport, food, tickets, and lodging if needed. 
The family owns one car (a 10-year-old sedan, reliable but uses a lot of fuel).

THE FAMILY

Alex (the Planner), 24, older sibling
- Works as an accountant, recently moved back home to save money.
- Organized, keeps a notes app full of lists, likes having a backup plan.
- Wants this trip to be a nice memory since the family rarely does things together anymore.
- Has a driver's license and can share driving with Dad.

Dad, 52
- Works long hours as a construction site supervisor, tired by the weekend.
- Careful with money; remembers when the family had financial struggles.
- Has a bad back, so long drives (over 3 hours) and hard mattresses are a problem.
- Secretly enjoys fishing and nature, though he rarely mentions it.
- Likes to tell stories from his youth.

Mia (Little Sister), 15
- In high school, very active on social media.
- Loves trying new food, taking aesthetic photos, and anything "trendy."
- Hates hiking, early mornings, and places without good mobile internet.
- Has a group project due Monday, so she needs a few hours on Sunday evening.
- Would rather spend time with friends but would never admit she enjoys family trips.
"""

chatHistory: list[ChatMessage] = [{"role":"assistant", "content": tripScenarioPrompt}]

def appendToChatHistoryFromLlmResponse(reply: FamilyMessage, speaker: str) -> None:
    chatHistory.append({"role":"assistant", "content": f"From {speaker} to {reply['to']}: {reply['content']}"})

judgeModel: str = "ollama_chat/qwen3:8b"
maxAttempts: int = 3
maxRounds: int = 100

judgeSystemPrompt: str = """
    You are a strict reviewer checking ONE reply in a family role-play chat about a weekend trip.
    The family members are exactly: "Alex", "Dad", "Mia".
    A reply is VALID only if ALL of these are true:
    1. It is ONE JSON object in this exact shape: {"to": "<name>", "content": "<text>"}
    2. "to" is exactly "Alex", "Dad" or "Mia", and never the speaker themself.
    3. "content" is a non-empty string.
    4. The content stays in the speaker's character and actually responds to the incoming message.
    Respond ONLY with a JSON object: {"valid": true or false, "reason": "<one short sentence>"}
"""


async def judgeResponse(speaker: str, rolePrompt: str, incoming: str, raw: str) -> JudgeVerdict:
    judgeInput: str = f"""/no_think
SPEAKER: {speaker}
SPEAKER'S ROLE:
{rolePrompt}

INCOMING MESSAGE THE SPEAKER IS ANSWERING:
{incoming}

SPEAKER'S REPLY TO CHECK:
{raw}
"""
    llmResponse = await acompletion(
        model=judgeModel,
        messages=[{"role": "system", "content": judgeSystemPrompt}, {"role": "user", "content": judgeInput}],
        response_format={"type": "json_object"},
    )
    verdict: JudgeVerdict = json.loads(llmResponse.choices[0].message.content)
    return verdict


async def askWithRetry(speaker: str, model: str, rolePrompt: str, messages: list[ChatMessage], incoming: str) -> FamilyMessage:
    for attempt in range(1, maxAttempts + 1):
        llmResponse = await acompletion(model=model, messages=messages, response_format={"type": "json_object"})
        realResponseAsString: str = llmResponse.choices[0].message.content
        try:
            realResponseAsJson: FamilyMessage = json.loads(realResponseAsString)
        except ValueError:
            console.print(f"[red]✗ {speaker} attempt {attempt}/{maxAttempts}: invalid JSON, retrying[/]")
            continue

        verdict: JudgeVerdict = await judgeResponse(speaker, rolePrompt, incoming, realResponseAsString)
        if verdict.get("valid"):
            console.print(f"[dim green]✓ {speaker} reply approved by qwen3:8b[/]")
            return realResponseAsJson

        console.print(f"[red]✗ {speaker} attempt {attempt}/{maxAttempts}: {verdict.get('reason')}, retrying[/]")

    raise RuntimeError(f"{speaker} failed to give a valid reply after {maxAttempts} attempts")


async def callAlex(content: str, sender: str) -> FamilyMessage:
    messages: list[ChatMessage] = [{"role":"system", "content": alexSystemPrompt}] + chatHistory + [{"role":"user", "content": content}]
    chatHistory.append({"role":"assistant", "content": f"From {sender} to Alex: {content}"})

    reply: FamilyMessage = await askWithRetry("Alex", "ollama_chat/qwen3:1.7b", alexSystemPrompt, messages, content)
    appendToChatHistoryFromLlmResponse(reply, "Alex")

    return reply

async def callMia(content: str, sender: str) -> FamilyMessage:
    messages: list[ChatMessage] = [{"role":"system", "content": miaSystemPrompt}] + chatHistory + [{"role":"user", "content": content}]
    chatHistory.append({"role":"assistant", "content": f"From {sender} to Mia: {content}"})

    reply: FamilyMessage = await askWithRetry("Mia", "ollama_chat/llama3.2", miaSystemPrompt, messages, content)
    appendToChatHistoryFromLlmResponse(reply, "Mia")

    return reply

async def callDad(content: str, sender: str) -> FamilyMessage:
    messages: list[ChatMessage] = [{"role":"system", "content": dadSystemPrompt}] + chatHistory + [{"role":"user", "content": content}]
    chatHistory.append({"role":"assistant", "content": f"From {sender} to Dad: {content}"})

    reply: FamilyMessage = await askWithRetry("Dad", "openai/gpt-4.1", dadSystemPrompt, messages, content)
    appendToChatHistoryFromLlmResponse(reply, "Dad")

    return reply

async def routeMessageToFamilyMember(fromWho: str, message: FamilyMessage) -> FamilyMessage:
    to: FamilyMember = message["to"]
    content: str = message["content"]
    res: FamilyMessage
    color: str = memberColors.get(fromWho, "white")
    console.print(Panel(
        Markdown(f"**{fromWho}** ➜ **{to}**\n\n{content}"),
        title=f"[bold {color}]{fromWho}[/]",
        subtitle=f"to [bold {memberColors.get(to, 'white')}]{to}[/]",
        border_style=color,
        expand=False,
    ))
    console.print()
    await asyncio.sleep(3)  # Simulate a delay for realism

    if to == "Alex":
        res = await callAlex(content,fromWho)
    elif to == "Dad":
        res = await callDad(content,fromWho)
    elif to == "Mia":
        res = await callMia(content,fromWho)
    else:
        raise ValueError(f"Unknown family member: {to}")

    return res

async def start() -> None:
    firstMessage: str = "Hey Alex. I think you are planner. what kind of plan do you have for this weekend trip?"
    fromWho: str = "System"
    message: FamilyMessage = {"to": "Alex", "content": firstMessage}

    for turn in range(1, maxRounds + 1):
        console.rule(f"[bold]Turn {turn}/{maxRounds}[/]")
        reply: FamilyMessage = await routeMessageToFamilyMember(fromWho, message)
        # The receiver answers, so they become the sender of the next message
        fromWho, message = message["to"], reply

    console.rule("[bold]Conversation finished[/]")



if __name__ == "__main__":
    asyncio.run(start())



    