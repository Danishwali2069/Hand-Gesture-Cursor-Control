import asyncio
from pathlib import Path

import websockets


SAVE_DIR = Path("received_files")
SAVE_DIR.mkdir(exist_ok=True)


async def handler(websocket):
    filename = await websocket.recv()
    data = await websocket.recv()
    out = SAVE_DIR / filename
    out.write_bytes(data)
    await websocket.send(f"saved:{out.name}")


async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, max_size=50 * 1024 * 1024):
        print("WebSocket file server on ws://0.0.0.0:8765")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
