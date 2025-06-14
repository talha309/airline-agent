from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import loadenv
import os
loadenv()
MONGO_URL=os.getenv("DATABASE_URL")
cient=AsyncIOMotorClient(MONGO_URL)
db= client["airline_db"]
chat_collection=db["chat_history"]
flight_collection=db["flight_history"]