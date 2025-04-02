from datetime import datetime
import asyncio
import argparse

import numpy
import websockets


async def send_coroutine(
    websocket: websockets.ClientConnection, int16_arr: numpy.ndarray, id
):
    chunk_size = 160
    for i in range(0, len(int16_arr), chunk_size):
        chunk = int16_arr[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = numpy.pad(chunk, (0, chunk_size - len(chunk)), "constant")
        await websocket.send(chunk.tobytes())
    blank = b"0" * chunk_size * 2
    for i in range(200):
        await websocket.send(blank)
    return datetime.now()

async def get_coroutine(websocket: websockets.ClientConnection, id):
    while True:
        result = await websocket.recv()
        if "speech transcript" in result:
            return datetime.now()


async def simulate_client(i: int, uri: str, data: numpy.ndarray, repeat: int):
    async with websockets.connect(uri, open_timeout=20) as websocket:
        for _ in range(repeat):
            try:
                await asyncio.gather(
                    send_coroutine(websocket, data, i),
                    get_coroutine(websocket, i),
                )
            except asyncio.CancelledError:
                pass


async def simulate_client_together(i, uri, data, repeat):
    async with websockets.connect(uri) as websocket:
        for _ in range(repeat):
            try:
                send = await send_coroutine(websocket, data, i)
                get = await get_coroutine(websocket, i)
                print(get - send)
            except asyncio.CancelledError:
                pass


def client_wrapper(i: int, uri: str, data: str, repeat: int):
    asyncio.run(simulate_client(i, uri, data, repeat))


async def run_all_clients(uri: str, data: numpy.ndarray, num_clients: int, repeat: int):
    tasks = [
        simulate_client_together(i, uri, data.copy(), repeat)
        for i in range(num_clients)
    ]
    await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser(description="Stress test for the server")
    parser.add_argument(
        "--uri",
        type=str,
        help="URI of the server",
        default="ws://localhost:8766/stt",
    )
    parser.add_argument(
        "--clients", type=int, help="Number of clients", default=1
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the data file",
        default="test/data/short_test_data.pcm",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of times to repeat the data",
        default=1,
    )
    args = parser.parse_args()
    with open(args.data, "rb") as file:
        data = file.read()
        int16_arr = numpy.frombuffer(data, dtype=numpy.int16)

    asyncio.run(run_all_clients(args.uri, int16_arr, args.clients, args.repeat))

    # threads = [
    #     threading.Thread(
    #         target=client_wrapper,
    #         args=(i, args.uri, int16_arr.copy(), args.repeat),
    #     )
    #     for i in range(args.clients)
    # ]
    # for thread in threads:
    #     thread.start()

    # for thread in threads:
    #     thread.join()


if __name__ == "__main__":
    main()
