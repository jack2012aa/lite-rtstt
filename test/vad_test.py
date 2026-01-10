import unittest

from lite_rtstt.stt.audiobuffer import AudioBuffer
from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.vad_client import MockVADClient, SileroClient, WebRTCClient
from test.audio_provider import from_int16_pcm, get_silence_audio


class MockVADClientTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.__results = [True, True, False, True, False]
        self.__mock_client = MockVADClient()
        await self.__mock_client.append_result(*self.__results)

    async def asyncTearDown(self):
        self.__mock_client = None

    async def test_is_active(self):
        with self.assertRaises(RuntimeError):
            await self.__mock_client.is_active(None)
        self.__mock_client.start()
        for expected in self.__results:
            actual = await self.__mock_client.is_active(None)
            self.assertEqual(expected, actual)
        self.__mock_client.close()
        with self.assertRaises(RuntimeError):
            await self.__mock_client.is_active(None)


class SileroClientTest(unittest.IsolatedAsyncioTestCase):

    __voice = from_int16_pcm("test/data/7s_i16.pcm", 30)
    __silence = get_silence_audio(7000)

    async def asyncSetUp(self):
        self.__config = STTConfig.default()
        self.__client = SileroClient(self.__config)

    async def asyncTearDown(self):
        self.__client.close()

    async def test_is_active(self):
        with self.assertRaises(RuntimeError):
            await self.__client.is_active(self.__silence)
        self.__client.start()
        actual = await self.__client.is_active(self.__silence)
        self.assertEqual(False, actual)
        actual = await self.__client.is_active(self.__voice)
        self.assertEqual(True, actual)
        self.__client.close()
        with self.assertRaises(RuntimeError):
            await self.__client.is_active(self.__silence)


class RTCClientTest(unittest.IsolatedAsyncioTestCase):

    __voice = AudioBuffer.from_bytes(from_int16_pcm("test/data/7s_i16.pcm", 30).get_chunk(100))
    __silence = get_silence_audio(30)

    async def asyncSetUp(self):
        self.__config = STTConfig.default()
        self.__client = WebRTCClient(self.__config)

    async def asyncTearDown(self):
        self.__client.close()

    async def test_is_active(self):
        with self.assertRaises(RuntimeError):
            await self.__client.is_active(self.__silence)
        self.__client.start()
        actual = await self.__client.is_active(self.__silence)
        self.assertEqual(False, actual)
        actual = await self.__client.is_active(self.__voice)
        self.assertEqual(True, actual)
        self.__client.close()
        with self.assertRaises(RuntimeError):
            await self.__client.is_active(self.__silence)

if __name__ == '__main__':
    unittest.main()
