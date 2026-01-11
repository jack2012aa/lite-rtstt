import asyncio
import os
import shutil
import unittest

from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.stt_client import MockSTTClient, WhisperClient
from test.utils import from_int16_pcm, get_silence_audio, assert_text_similar


class MockSTTClientTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.__client = MockSTTClient()
        self.__results = ("I", "am", "the", "storm", "that", "is", "approaching")
        await self.__client.append_results(*self.__results)

    async def asyncTearDown(self):
        self.__client.close()
        self.__client = None

    async def test_transcribe(self):
        with self.assertRaises(RuntimeError):
            await self.__client.transcribe(None)
        self.__client.start()
        for expected in self.__results:
            self.assertEqual(expected, await self.__client.transcribe(None))
        self.__client.close()
        with self.assertRaises(RuntimeError):
            await self.__client.transcribe(None)


class WhisperClientTest(unittest.IsolatedAsyncioTestCase):

    __voice = from_int16_pcm("test/data/7s_i16.pcm")
    __silence = get_silence_audio(7000)

    async def asyncSetUp(self):
        self.__config = STTConfig.default()
        self.__temp_dir = "test_temp"
        os.mkdir(self.__temp_dir)
        self.__client = WhisperClient(self.__config, self.__temp_dir)

    async def asyncTearDown(self):
        self.__client.close()
        self.__client = None
        shutil.rmtree(self.__temp_dir)

    async def test_transcribe(self):
        with self.assertRaises(RuntimeError):
            await self.__client.transcribe(self.__silence)
        self.__client.start()
        # The model is downloaded. Should return fast.
        async with asyncio.timeout(1):
            actual = await self.__client.transcribe(self.__silence)
            self.assertEqual("", actual)
        async with asyncio.timeout(1):
            expected = "You are given an integer matrix grid and an array queries of size k."
            actual = await self.__client.transcribe(self.__voice)
            assert_text_similar(self, expected, actual)
        self.__client.close()
        with self.assertRaises(RuntimeError):
            await self.__client.transcribe(self.__silence)


if __name__ == '__main__':
    unittest.main()
