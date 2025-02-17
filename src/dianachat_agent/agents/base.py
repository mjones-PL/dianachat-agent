from livekit import agents
from livekit.agents import RoomServiceClient

class BaseAgent(agents.BaseAgent):
    def __init__(self, room_client: RoomServiceClient):
        super().__init__()
        self.room_client = room_client

    async def on_connect(self):
        """Called when agent connects to room"""
        print(f"Agent connected to room: {self.room_client.room.name}")

    async def on_disconnect(self):
        """Called when agent disconnects from room"""
        print(f"Agent disconnected from room: {self.room_client.room.name}")

    async def on_error(self, error: Exception):
        """Called when an error occurs"""
        print(f"Error occurred: {error}")
