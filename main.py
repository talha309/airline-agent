from __future__ import annotations as _annotations

import asyncio
import random
import uuid
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import asyncio

from agents_setup import triage_agent, AirlineAgentContext, Runner, ItemHelpers, MessageOutputItem, HandoffOutputItem, ToolCallItem, ToolCallOutputItem
from db import chat_collection, flight_collection

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
    trace,
    ChatModel,
    ChatMessage,
)

from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from db import chat_collection, flight_collection 

app = FastAPI()
### LOAD ENV ###
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


### GEMINI WRAPPER ###
class GeminiChatModel(ChatModel):
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def complete_chat(self, messages: list[ChatMessage], **kwargs) -> ChatMessage:
        prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
        response = self.model.generate_content(prompt)
        return ChatMessage(role="assistant", content=response.text)


gemini_model = GeminiChatModel(api_key=gemini_api_key)


### CONTEXT ###
class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


### TOOLS ###
@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    if "bag" in question or "baggage" in question:
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif "seats" in question or "plane" in question:
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
        )
    elif "wifi" in question:
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


### HOOK ###
async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


### AGENTS ###
faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    model=gemini_model,
    tools=[faq_lookup_tool],
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent.""",
    model=gemini_model,
    tools=[update_seat],
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=f"{RECOMMENDED_PROMPT_PREFIX} You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents.",
    model=gemini_model,
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


### RUNNER ###
async def main():
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()
    conversation_id = uuid.uuid4().hex[:16]

    while True:
        user_input = input("Enter your message: ")
        with trace("Customer service", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

            for new_item in result.new_items:
                agent_name = new_item.agent.name
                if isinstance(new_item, MessageOutputItem):
                    print(f"{agent_name}: {ItemHelpers.text_message_output(new_item)}")
                elif isinstance(new_item, HandoffOutputItem):
                    print(f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")
                elif isinstance(new_item, ToolCallItem):
                    print(f"{agent_name}: Calling a tool")
                elif isinstance(new_item, ToolCallOutputItem):
                    print(f"{agent_name}: Tool call output: {new_item.output}")
                else:
                    print(f"{agent_name}: Skipping item: {new_item.__class__.__name__}")
            input_items = result.to_input_list()
            current_agent = result.last_agent
        
 # 👈 Import MongoDB collections

async def main():
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()

    conversation_id = uuid.uuid4().hex[:16]
    session_id = str(uuid.uuid4())[:8]  # Unique ID for user session

    while True:
        user_input = input("Enter your message: ")

        # Store user input in MongoDB
        await chat_collection.insert_one({
            "session_id": session_id,
            "role": "user",
            "message": user_input
        })

        with trace("Customer service", group_id=conversation_id):
            input_items.append({"content": user_input, "role": "user"})
            result = await Runner.run(current_agent, input_items, context=context)

        for new_item in result.new_items:
            agent_name = new_item.agent.name

            if isinstance(new_item, MessageOutputItem):
                message = ItemHelpers.text_message_output(new_item)
                print(f"{agent_name}: {message}")

                # Save response to chat history in MongoDB
                await chat_collection.insert_one({
                    "session_id": session_id,
                    "role": agent_name,
                    "message": message
                })

            elif isinstance(new_item, HandoffOutputItem):
                print(f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}")

            elif isinstance(new_item, ToolCallItem):
                print(f"{agent_name}: Calling a tool")

            elif isinstance(new_item, ToolCallOutputItem):
                print(f"{agent_name}: Tool call output: {new_item.output}")

            else:
                print(f"{agent_name}: Skipping item: {new_item._class.name_}")

        # Save flight booking info if all values present
        if all([
            context.confirmation_number,
            context.flight_number,
            context.seat_number,
        ]):
            flight_doc = {
                "session_id": session_id,
                "confirmation_number": context.confirmation_number,
                "flight_number": context.flight_number,
                "seat_number": context.seat_number
            }
            # Check if already stored
            existing = await flight_collection.find_one(flight_doc)
            if not existing:
                await flight_collection.insert_one(flight_doc)

        input_items = result.to_input_list()
        current_agent = result.last_agent
    


# Session state
agent_state = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())[:8]
    user_input = request.message

    # Restore or create state
    if session_id not in agent_state:
        agent_state[session_id] = {
            "agent": triage_agent,
            "input_items": [],
            "context": AirlineAgentContext()
        }

    state = agent_state[session_id]
    agent = state["agent"]
    input_items = state["input_items"]
    context = state["context"]
    conversation_id = session_id

    await chat_collection.insert_one({
        "session_id": session_id,
        "role": "user",
        "message": user_input
    })

    input_items.append({"content": user_input, "role": "user"})
    result = await Runner.run(agent, input_items, context=context)

    messages = []
    for new_item in result.new_items:
        agent_name = new_item.agent.name

        if isinstance(new_item, MessageOutputItem):
            msg = ItemHelpers.text_message_output(new_item)
            messages.append({"role": agent_name, "message": msg})
            await chat_collection.insert_one({
                "session_id": session_id,
                "role": agent_name,
                "message": msg
            })

        elif isinstance(new_item, HandoffOutputItem):
            messages.append({
                "role": "system",
                "message": f"Handed off from {new_item.source_agent.name} to {new_item.target_agent.name}"
            })

        elif isinstance(new_item, ToolCallItem):
            messages.append({"role": agent_name, "message": "Calling a tool"})

        elif isinstance(new_item, ToolCallOutputItem):
            messages.append({"role": agent_name, "message": f"Tool call output: {new_item.output}"})

    if all([
        context.confirmation_number,
        context.flight_number,
        context.seat_number,
    ]):
        flight_doc = {
            "session_id": session_id,
            "confirmation_number": context.confirmation_number,
            "flight_number": context.flight_number,
            "seat_number": context.seat_number
        }
        existing = await flight_collection.find_one(flight_doc)
        if not existing:
            await flight_collection.insert_one(flight_doc)

    # Save updated state
    agent_state[session_id] = {
        "agent": result.last_agent,
        "input_items": result.to_input_list(),
        "context": context
    }

    return JSONResponse({"session_id": session_id, "responses": messages})


if __name__ == "__main__":
    asyncio.run(main())
