from datetime import datetime
import asyncio
import argparse
import threading
from time import sleep
import multiprocessing

import numpy
import websockets


async def send_coroutine(
    websocket: websockets.ClientConnection, int16_arr: numpy.ndarray
):
    for i in range(0, len(int16_arr), 128):
        tail = min(len(int16_arr), i + 128)
        await websocket.send(int16_arr[i:tail].tobytes())
    print("Sent data")
    # Send an empty byte to indicate the end of the audio stream.
    empty_arr = numpy.array([0] * 128, dtype=numpy.int16)
    while True:
        await websocket.send(empty_arr.tobytes())
        await asyncio.sleep(0.1)


async def get_coroutine(websocket: websockets.ClientConnection):
    while True:
        async with asyncio.timeout(10):
            result = await websocket.recv()
            print("Received data: ", result)
            if "speech transcript" in result:
                raise asyncio.CancelledError()


async def simulate_client(i: int, uri: str, data: numpy.ndarray, repeat: int):
    connection_time = datetime.now()
    async with websockets.connect(uri, open_timeout=20) as websocket:
        starting_time = datetime.now()
        print(f"Client {i} connected to the server at {starting_time}")
        for _ in range(repeat):
            try:
                await asyncio.gather(
                    send_coroutine(websocket, data),
                    get_coroutine(websocket),
                )
            except asyncio.CancelledError:
                pass
        end_time = datetime.now()
        print(f"Client {i} finished at {end_time}")
    print(
        f"Client {i} time logs: {connection_time}, {starting_time}, {end_time}, {end_time - connection_time}"
    )


def client_wrapper(i: int, uri: str, data: str, repeat: int):
    asyncio.run(simulate_client(i, uri, data, repeat))


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
    processes = [
        multiprocessing.Process(
            target=client_wrapper, args=(i, args.uri, int16_arr, args.repeat)
        )
        for i in range(args.clients)
    ]
    for process in processes:
        process.start()
        sleep(2)
    for process in processes:
        process.join()
    # threads = [threading.Thread(target=client_wrapper, args=(i, args.uri)) for i in range(args.clients)]
    # for thread in threads:
    #     thread.start()
    #     sleep(2)

    # for thread in threads:
    #     thread.join()


if __name__ == "__main__":
    main()
