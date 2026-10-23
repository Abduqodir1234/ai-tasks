import code
from contextlib import closing
from contextvars import ContextVar
from datetime import date, datetime
import json
import logging
import time
import sqlite3
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
import gradio as gr
from litellm import Message, completion
from openai import OpenAI
from io import BytesIO
import base64
from PIL import Image
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("airline-assistant")

openAi = OpenAI()

qwenModel: str = "ollama_chat/qwen3:8b"
qwenModelApiBase = 'http://127.0.0.1:11434'
COMPANY_NAME = "Airsoft Airway"
WEBSITE = "airsoftairway.example"
SALES_CHANNELS = f"website {WEBSITE}, the Airsoft Airway mobile app (iOS/Android), sales offices in Tashkent and Samarkand airports, or by phone +998 71 200 00 00"
SUPPORT_CONTACT = "+998 71 200 00 00 (24/7) or support@airsoftairway.example"
AVAILABILITY_TOOL = "the ticketListByCity tool"

FARE_TYPES = """
  - Light: seat only, 1 hand luggage item up to 8 kg, no checked baggage, seat selection paid.
  - Standard: 23 kg checked baggage + 8 kg hand luggage, free standard seat selection, meal included.
  - Flex: 2 x 23 kg checked baggage + 10 kg hand luggage, free seat selection incl. extra legroom, priority check-in and boarding, meal included."""

BAGGAGE_RULES = """
  - Hand luggage: max 55 x 40 x 20 cm, plus one personal item (handbag or laptop bag).
  - Checked baggage: each piece max 158 cm (length + width + height).
  - Overweight 23–32 kg: 30 USD per piece. Extra piece: 50 USD.
  - Infants get 10 kg checked baggage and one folding stroller free."""

CHILD_INFANT_RULES = """
  - Infant (under 2 years, no own seat): 10% of the adult fare, one infant per adult.
  - Child (2–11 years, own seat): 75% of the adult fare.
  - 12 years and older: adult fare.
  - Children under 12 must travel with an adult (16+)."""

CHANGE_REFUND_RULES = """
  - Light: no changes, no refunds (only taxes are refunded).
  - Standard: changes allowed for 40 USD + fare difference up to 24 hours before departure; refund with 60 USD fee.
  - Flex: free changes (fare difference only) up to 3 hours before departure; full refund up to 24 hours before departure."""

DOCUMENTS = "a valid passport (ID card for domestic flights) for every passenger; birth certificate for children without a passport"

CHECK_IN_RULES = """
  - Online check-in opens 24 hours and closes 3 hours before departure (on the website or app).
  - Airport check-in opens 3 hours and closes 45 minutes before departure (international: 60 minutes).
  - Boarding gate closes 20 minutes before departure."""

sysPrompt = f"""
# Role
Today's date is {date.today().isoformat()}.
You are the ticket information assistant for **{COMPANY_NAME}**. You do two things only:
1. Check whether tickets are available for a flight.
2. Answer customers' questions about tickets.

You do NOT book tickets, take payments, or change or cancel bookings. When a customer wants to buy, send them to {SALES_CHANNELS}.

# 1. Checking ticket availability
Ask for:
- From where (city or airport)
- To where
- Date of travel (and return date, if needed)
- Number of passengers (adults / children / infants), if they want to know for a group

Then check using {AVAILABILITY_TOOL} and reply with:
- Whether tickets are available: **Yes / No**
- If yes: available flights with date, departure and arrival time, and starting price (if the system gives it)
- If no: the nearest dates or flights that do have tickets, if the system shows them

Never guess availability, prices, or schedules. If the system doesn't return data or is unavailable, say: "I can't check right now. Please try again later or contact {SUPPORT_CONTACT}."

# 2. Questions about tickets
You can answer only these, using the facts below:
- Fare types and what each includes
- Baggage and hand luggage allowance by fare
- Ticket prices for children and infants
- Whether a ticket can be changed or refunded (the rules by fare, not processing an actual change)
- What documents are needed to buy a ticket
- Where and how to buy a ticket
- Check-in times and rules

Facts:
- Fare types:{FARE_TYPES}
- Baggage rules:{BAGGAGE_RULES}
- Child / infant rules:{CHILD_INFANT_RULES}
- Change / refund rules:{CHANGE_REFUND_RULES}
- Documents: {DOCUMENTS}
- Where to buy: {SALES_CHANNELS}
- Check-in:{CHECK_IN_RULES}
- Support contact: {SUPPORT_CONTACT}

If the answer isn't in these facts, say you don't have that information and give the support contact. Do not invent answers.

# Off-topic requests
For anything not about ticket availability or ticket questions (hotels, visas, other airlines, complaints, lost baggage, general chat), reply:
"I can only help with checking ticket availability and questions about tickets. For other questions, please contact {SUPPORT_CONTACT}."

# Rules
- Never ask for passport numbers, card details, or other personal data. You don't need them.
- Don't promise that a ticket will still be available later; availability can change at any moment.
- Don't invent urgency or discounts.

# Language and style
- Reply in the customer's language: Uzbek (Latin or Cyrillic), Russian, or English.
- Keep answers short and clear. Ask only for what's missing.
- End with a helpful next step, such as offering to check another date or pointing to where to buy.

# Examples
**Customer:** 20-noyabrga Toshkentdan Moskvaga bilet bormi?
**Assistant:** [checks system] Ha, 20-noyabrga bilet bor:
- 08:30 → 12:15, narxi 210 USD dan
- 19:40 → 23:25, narxi 185 USD dan
Sotib olish uchun: {WEBSITE}.

**Customer:** Is there a ticket to Dubai on December 3?
**Assistant:** [checks system] Unfortunately, there are no tickets left for December 3. The nearest available dates are December 2 and December 5. Want me to check one of those?

**Customer:** How many kg of baggage can I take?
**Assistant:** It depends on the fare: Light — hand luggage only, 8 kg; Standard — 23 kg checked + 8 kg hand luggage; Flex — 2 x 23 kg checked + 10 kg hand luggage. Which fare are you looking at?

**Customer:** Can you book a hotel for me?
**Assistant:** Sorry, I can only help with checking ticket availability and ticket questions. For other questions, please contact {SUPPORT_CONTACT}.
"""


class GradioChatHistoryContent(TypedDict):
    text: str
    type: str

class GradioChatHistory(TypedDict):
    role: str
    metadata: None
    content: list[GradioChatHistoryContent]
    options: None

class LiteLLMMessages(TypedDict):
    role: str
    content: str

class TicketListByCityParams(TypedDict, total=False):
    from_city: str
    to_city: str
    date: str
    from_date: str
    to_date: str
    offset: int


DB_PATH = Path(__file__).with_name("airline.db")
PAGE_SIZE = 10

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def gradioToLitellm(history: list[GradioChatHistory]) -> list[LiteLLMMessages]:
    messages: list[LiteLLMMessages] = []
    for msg in history:
        texts = [
            part["text"]
            for part in msg["content"]
            if part.get("type") == "text" and part.get("text")
        ]
        if not texts:
            continue  # skip file-only or empty messages
        messages.append({"role": msg["role"], "content": "\n".join(texts)})
    return messages

def ticketListByCity(params: TicketListByCityParams):
    fromCity = (params.get("from_city") or "").strip()
    toCity = (params.get("to_city") or "").strip()
    if not fromCity or not toCity:
        return {"error": "from_city and to_city are required"}

    where = ["from_city = ? COLLATE NOCASE", "to_city = ? COLLATE NOCASE"]
    args: list = [fromCity, toCity]

    dates: dict[str, str] = {}
    for key in ("date", "from_date", "to_date"):
        value = (params.get(key) or "").strip()
        if not value:
            continue
        try:
            date.fromisoformat(value)
        except ValueError:
            return {"error": f"{key} must be in YYYY-MM-DD format"}
        dates[key] = value

    # An exact date is the same as a one-day range
    fromDate = dates.get("date") or dates.get("from_date")
    toDate = dates.get("date") or dates.get("to_date")
    if fromDate and toDate and fromDate > toDate:
        return {"error": "from_date must be on or before to_date"}

    if fromDate:
        where.append("substr(departure_time, 1, 10) >= ?")
        args.append(fromDate)
    else:
        # Without a start date, only show flights that haven't departed yet
        where.append("departure_time >= ?")
        args.append(datetime.now().isoformat(timespec="seconds"))
    if toDate:
        where.append("substr(departure_time, 1, 10) <= ?")
        args.append(toDate)

    offset = max(int(params.get("offset") or 0), 0)
    whereSql = " AND ".join(where)

    with closing(get_db()) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM flights WHERE {whereSql}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM flights WHERE {whereSql} ORDER BY departure_time LIMIT ? OFFSET ?",
            args + [PAGE_SIZE, offset],
        ).fetchall()

    logger.info("DB query %s -> %s dates=%s..%s offset=%d: %d of %d flights", fromCity, toCity, fromDate or "now", toDate or "any", offset, len(rows), total)

    return {
        "total": total,
        "offset": offset,
        "limit": PAGE_SIZE,
        "flights": [dict(row) for row in rows],
    }

llmToolsFunctions = {
    ticketListByCity.__name__: ticketListByCity,
}
TOOLS_LIST_TO_LLM = [
    {
        "type": "function",
        "function": {
            "name": ticketListByCity.__name__,
            "description": (
                "Search flights between two cities, sorted by departure time. "
                f"Returns at most {PAGE_SIZE} flights per call plus the total count; "
                "use offset to get the next page. available_seats = 0 means sold out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_city": {
                        "type": "string",
                        "description": "Departure city in English, e.g. Tashkent"
                    },
                    "to_city": {
                        "type": "string",
                        "description": "Arrival city in English, e.g. Istanbul"
                    },
                    "date": {
                        "type": "string",
                        "description": "Exact travel date in YYYY-MM-DD format. Use it when the user names one day; don't combine with from_date/to_date"
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Earliest departure date (inclusive) in YYYY-MM-DD format, e.g. for 'after Nov 10' or the start of a date range"
                    },
                    "to_date": {
                        "type": "string",
                        "description": "Latest departure date (inclusive) in YYYY-MM-DD format, e.g. for 'before Dec 1' or the end of a date range. Omit all date fields to list all upcoming flights"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of flights to skip for pagination (default 0)"
                    }
                },
                "required": ["from_city", "to_city"]
            }
        }
    }
]

def handleToolCall(message:  Message):
    responses = []
    for toolCall in message.tool_calls or []:
        funcName = toolCall.function.name
        # Every tool call needs a reply, so errors are sent back to the model instead of being skipped
        if funcName not in llmToolsFunctions:
            res = {"error": f"Unknown function: {funcName}"}
        else:
            try:
                funcParams = json.loads(toolCall.function.arguments or "{}")
                res = llmToolsFunctions[funcName](funcParams)
            except (ValueError, TypeError) as e:
                res = {"error": f"Invalid arguments: {e}"}

        logger.info("Tool call %s(%s)", funcName, toolCall.function.arguments)
        if "error" in res:
            logger.warning("Tool %s error: %s", funcName, res["error"])

        responses.append({
            "role": "tool",
            "content": json.dumps(res, ensure_ascii=False),
            "tool_call_id": toolCall.id
        })

    return responses

def getDestinationAddress(messages: list[LiteLLMMessages]):
    userPrompt: str = """
        You are a destination extraction assistant for an airline booking chat.
        Find the destination city the user is CURRENTLY interested in, and return it only if
        the assistant has confirmed that flights to that city are available.

        How to read the conversation:
        1. The user may ask about several cities one after another. Only the MOST RECENT destination
        matters. Earlier cities are forgotten once the user moves to a new one.
        2. Short replies like "yes", "yep", "ok", "book it", "that one", "this date" do not name a city.
        They refer to the destination in the assistant's previous message.
        3. In "X to Y" or "from X to Y", Y is the destination. Never return the origin city.
        4. Check the assistant's messages about that most recent destination:
        - If they list flights, times, prices or available seats → available.
        - If they say no flights, sold out or not available → return null.
        - If the assistant has not answered about that city yet → return null.
        5. The user only asking about a city does not mean it is available.

        Return the city's full address in English as "City, Region/State, Country"
        (for example Bukhara → "Bukhara, Bukhara Region, Uzbekistan",
        Istanbul → "Istanbul, Istanbul Province, Turkey", Dubai → "Dubai, Emirate of Dubai, United Arab Emirates").
        If you are not sure of the region, use "City, Country".

        Examples:
        - Istanbul (flights listed) → Paris (no flights) → Bukhara (flights listed) → user: "yep"
        → {"city": "Bukhara", "address": "Bukhara, Bukhara Region, Uzbekistan"}
        - Istanbul (flights listed) → user: "what about Paris" → assistant: no flights to Paris
        → {"city": null, "address": null}
        - Bukhara (flights listed) → user: "can you tell me Tashkent to Bukhara on 2026-10-30"
        → assistant: flight on that date → {"city": "Bukhara", "address": "Bukhara, Bukhara Region, Uzbekistan"}
        - user: "flights to Rome?" (assistant has not answered yet)
        → {"city": null, "address": null}

        Output ONLY valid JSON, no explanation:
        {"city": "CITY_NAME", "address": "City, Region, Country"}
        or
        {"city": null, "address": null}
        """
    messages.append({"role": "user", "content": userPrompt})
    cityLlmResAsString = completion(model=qwenModel,messages=messages, api_base=qwenModelApiBase,tools = TOOLS_LIST_TO_LLM)
    res = json.loads(cityLlmResAsString.choices[0].message.content)
    logger.info("Destination extracted: city=%s address=%s", res.get("city"), res.get("address"))
    return res.get("address")

def chatWithLLM(prompt: str,gradioHistory: list[GradioChatHistory]):
    image = None
    history = gradioToLitellm(gradioHistory)
    currentMessage = {"role":"user", "content":prompt}
    messages = [{"role": "system", "content": sysPrompt}] + history + [currentMessage]
    logger.info("User prompt (%d history msgs): %s", len(history), prompt)

    llmRes= completion(messages = messages, model=qwenModel, api_base=qwenModelApiBase,tools=TOOLS_LIST_TO_LLM)

    toolRound = 0
    while llmRes.choices[0].finish_reason == 'tool_calls':
        toolRound += 1
        message = llmRes.choices[0].message
        logger.info("LLM requested %d tool call(s), round %d", len(message.tool_calls or []), toolRound)
        toolRes = handleToolCall(message)
        messages.append(message)
        messages.extend(toolRes)
        llmRes = completion(model=qwenModel,messages=messages, api_base=qwenModelApiBase,tools = TOOLS_LIST_TO_LLM)

    reply = llmRes.choices[0].message.content or ""
    logger.info("LLM reply (%d chars): %s", len(reply), reply[:150].replace("\n", " "))
    messages.append(llmRes.choices[0].message)

    cityAddress = getDestinationAddress(messages)

    if cityAddress:
        image = generateImageForCity(cityAddress)

    audio = convertTextToVoice(reply) if reply else None

    return reply, image, audio

def convertTextToVoice(text: str):
    logger.info("Generating voice (%d chars)", len(text))
    startedAt = time.perf_counter()
    res = openAi.audio.speech.create(model="tts-1",input=text,voice="ash")
    logger.info("Voice generated in %.1fs", time.perf_counter() - startedAt)

    return res.content

def generateImageForCity(cityAddress: str):
    prompt = f"""
        A cinematic travel advertisement photo of {cityAddress}.
        Show the most famous, instantly recognizable landmark of this exact city in the foreground, with a typical view of the city behind it.
        Golden hour, warm inviting light. A modern passenger airplane flying high in the sky, small and elegant, not covering the landmark.
        Style: ultra-realistic photography, vibrant natural colors, professional airline marketing look.
        Composition: landmark on the right two-thirds, clean open sky on the left third for text overlay.
        Aspect ratio: 16:9. No text, no letters, no logos, no watermarks.
    """
    logger.info("Generating image for %s", cityAddress)
    startedAt = time.perf_counter()
    llmRes = openAi.images.generate(model="gpt-image-2",prompt=prompt,n=1,size="1024x1024")
    logger.info("Image for %s generated in %.1fs", cityAddress, time.perf_counter() - startedAt)

    imageAsBase64 = llmRes.data[0].b64_json
    image_data = base64.b64decode(imageAsBase64)
    return Image.open(BytesIO(image_data))

with gr.Blocks(title="Airsoft AI Assistant") as ui:
    with gr.Row():
        with gr.Column(scale=2):
            cityImage = gr.Image(label="Destination", render=False, height=500)
            replyAudio = gr.Audio(label="Voice reply", render=False, autoplay=True, interactive=False)
            # chatWithLLM returns (message, image, audio); image and audio go to their blocks via additional_outputs
            gr.ChatInterface(fn=chatWithLLM, title="Airsoft AI Assistant", additional_outputs=[cityImage, replyAudio])
        with gr.Column(scale=1):
            cityImage.render()
            replyAudio.render()

ui.launch(inbrowser=True, auth=("test","test"))
