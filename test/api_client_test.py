import asyncio
import unittest
from datetime import datetime

import numpy
import dotenv

from service.speechmatics_api_client import SpeechmaticsClient
from service.aws_api_client import AWSApiClient

def print_text(text: str):
    print(f"Received transcript at {datetime.now().strftime('%H:%M:%S')}: {text}")

def print_start():
    print("Recording started")

def print_stop():
    print("Recording stopped")


class MyTestCase(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.path = "./test/data/42s_i16.pcm"
        self.chunk_size = 160  # 10 ms at 16 kHz
        self.sample_rate = 16000  # 16 kHz

    def tearDown(self):
        pass

    async def test_single_connection(self):

        client = SpeechmaticsClient(print_text, print_start, print_stop, 0.7)
        with open(self.path, "rb") as file:
            data = file.read()
            int16_arr = numpy.frombuffer(data, dtype=numpy.int16)
            for i in range(0, len(int16_arr), self.chunk_size):
                chunk = int16_arr[i : i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    chunk = numpy.pad(chunk, (0, self.chunk_size - len(chunk)), "constant")
                client.feed(chunk.tobytes())
                await asyncio.sleep(self.chunk_size / self.sample_rate)
        print(f"Finished feeding data at {datetime.now().strftime('%H:%M:%S')}")
        # Send some empty chunks to ensure the client processes all data
        for _ in range(self.sample_rate // self.chunk_size * 3):  # Two seconds of silence
            client.feed(b"\x00" * self.chunk_size)
            await asyncio.sleep(self.chunk_size / self.sample_rate)

        """
        Output: 
            You are given an integer matrix grid and an array queries of size k . Find an array answer of size k such that for each integer queries a, user in the top left cell of the matrix and repeat the following process . If queries are is strictly greater than the value of the current cell that you are in, then you get one point. If it is first time visiting the cell and you can move to any adjacent cell in all four directions. Otherwise you do not get any points and you end this process 
            
            You are given an integer matrix grid and an array queries of size k . Find an array answer of size k. Such sets for each integer queries a , you start in the top left cell of the matrix and repeat the following process . If queries I is strictly greater than the value of the current cell that you are in , then you get one point. If it is first time visiting the cell and you can move to any adjacent cell in all four directions. Otherwise you do not get any points and you end this process 
        """

    async def test_aws_single_connection(self):
        client = AWSApiClient(print_text, print_start, print_stop, 1)
        await client.run()
        with open(self.path, "rb") as file:
            data = file.read()
            int16_arr = numpy.frombuffer(data, dtype=numpy.int16)
            for i in range(0, len(int16_arr), self.chunk_size):
                chunk = int16_arr[i : i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    chunk = numpy.pad(chunk, (0, self.chunk_size - len(chunk)), "constant")
                await client.feed(chunk.tobytes())
                await asyncio.sleep(self.chunk_size / self.sample_rate)
        print(f"Finished feeding data at {datetime.now().strftime('%H:%M:%S')}")
        # Send some empty chunks to ensure the client processes all data
        for _ in range(self.sample_rate // self.chunk_size * 3):
            await client.feed(b"\x00" * self.chunk_size)
            await asyncio.sleep(self.chunk_size / self.sample_rate)
        await client.close()

        """
        Output: 
            You're given an integer matrics grid and an array queries of size k. Find an array answer of set case such sets for each integer queries I you starts in the top left cell of the matrix and repeats the following process.

            If query's eye is strictly greater than the value of the current cell that you're in, then you get 1 point if it is first time visiting this cell, and you can move to any adjacent cell in all four directions.
            
            Otherwise, you do not get any points and you and this process.
        """


if __name__ == "__main__":
    dotenv.load_dotenv()
    unittest.main()
