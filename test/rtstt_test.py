import asyncio
import os
import shutil
import unittest
from dataclasses import replace

from lite_rtstt.stt.config import STTConfig
from lite_rtstt.stt.event import EventFactory, StartSpeakingEvent, StopSpeakingEvent, TextEvent
from lite_rtstt.stt.rtstt_client import MockRTSTTClient, ThreeLayerRTSTTClient
from lite_rtstt.stt.stt_client import MockSTTClient, WhisperClient
from lite_rtstt.stt.vad_client import MockVADClient, WebRTCClient, SileroClient
from test.utils import get_silence_audio, assert_text_similar


class MockRTSTTClientTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.__client = MockRTSTTClient()
        self.__results = (EventFactory.start_speaking_event(), None, None, None, EventFactory.stop_speaking_event(), EventFactory.text_event("end"))

    async def asyncTearDown(self):
        self.__client.close()

    async def test_unsafe_connect_disconnect(self):
        with self.assertRaises(RuntimeError):
            self.__client.connect()
        with self.assertRaises(RuntimeError):
            self.__client.disconnect(0)
        self.__client.start()
        self.__client.close()
        with self.assertRaises(RuntimeError):
            self.__client.connect()

    async def test_feed(self):
        with self.assertRaises(RuntimeError):
            await self.__client.feed(0, None)
        self.__client.start()
        q, id = self.__client.connect()
        with self.assertRaises(KeyError):
            await self.__client.feed(id + 1, None)
        await self.__client.append_results(id, *self.__results)
        for expected in self.__results:
            await self.__client.feed(id, None)
            async with asyncio.timeout(0.1):
                actual = await q.get()
                self.assertEqual(expected, actual)
        self.__client.disconnect(id)
        with self.assertRaises(KeyError):
            await self.__client.feed(id, None)
        self.__client.close()
        with self.assertRaises(RuntimeError):
            await self.__client.feed(0, None)


class ThreeLayerRTSTTClientUnitTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.__first_vad = MockVADClient()
        self.__second_vad = MockVADClient()
        self.__stt = MockSTTClient()
        self.__pre_second_vad = 3
        self.__silence_pre_stt = 2
        self.__config = replace(STTConfig.default(), **{
            "duration_time_ms": 30 * self.__silence_pre_stt,
            "active_to_detection_ms": 30 * self.__pre_second_vad,
        })
        self.__client = ThreeLayerRTSTTClient(self.__config, self.__first_vad, self.__second_vad, self.__stt)

    async def asyncTearDown(self):
        self.__client.close()

    async def test_unsafe_connect_disconnect(self):
        with self.assertRaises(RuntimeError):
            self.__client.connect()
        with self.assertRaises(RuntimeError):
            self.__client.disconnect(0)
        self.__client.start()
        self.__client.close()
        with self.assertRaises(RuntimeError):
            self.__client.connect()

    async def test_feed(self):
        silence = get_silence_audio(30).to_bytes()
        with self.assertRaises(RuntimeError):
            await self.__client.feed(0, silence)
        self.__client.start()
        q, id = self.__client.connect()
        with self.assertRaises(KeyError):
            await self.__client.feed(id + 1, silence)

        # Noise
        first_vad_results = [True]
        second_vad_results = [False]
        await self.__first_vad.append_results(*first_vad_results)
        await self.__second_vad.append_results(*second_vad_results)
        for _ in range(self.__pre_second_vad):
            await self.__client.feed(id, silence)
        with self.assertRaises(TimeoutError):
            async with asyncio.timeout(0.1):
                await q.get()

        # Speaking
        first_vad_results = [True, False, True, False, False]
        second_vad_results = [True]
        stt_results = ["Hello there."]
        await self.__first_vad.append_results(*first_vad_results)
        await self.__second_vad.append_results(*second_vad_results)
        await self.__stt.append_results(*stt_results)

        for _ in range(7):
            await self.__client.feed(id, silence)

        async with asyncio.timeout(0.1):
            event = await q.get()
            self.assertIsInstance(event, StartSpeakingEvent)
        async with asyncio.timeout(0.1):
            event = await q.get()
            self.assertIsInstance(event, StopSpeakingEvent)
        async with asyncio.timeout(0.1):
            event = await q.get()
            self.assertIsInstance(event, TextEvent)
            self.assertEqual(stt_results[0], event.text)

        self.__client.disconnect(id)
        with self.assertRaises(KeyError):
            await self.__client.feed(id, silence)
        self.__client.close()
        with self.assertRaises(RuntimeError):
            await self.__client.feed(0, silence)


class ThreeLayerRTSTTClientIntegrationTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.__model_dir = "test_temp_model"
        if not os.path.exists(self.__model_dir):
            os.makedirs(self.__model_dir)

        self.__config = replace(STTConfig.default(), **{
            "whisper_model": "base",
            "vad_threads": 1
        })

        self.__rtc_vad = WebRTCClient(self.__config)
        self.__silero_vad = SileroClient(self.__config)
        self.__whisper_stt = WhisperClient(self.__config, download_root=self.__model_dir)

        self.__client = ThreeLayerRTSTTClient(
            self.__config,
            self.__rtc_vad,
            self.__silero_vad,
            self.__whisper_stt
        )

        self.__client.start()

    async def asyncTearDown(self):
        self.__client.close()
        if os.path.exists(self.__model_dir):
            shutil.rmtree(self.__model_dir)

    async def test_full_pipeline_recognition(self):
        pcm_path = "test/data/7s_i16.pcm"
        if not os.path.exists(pcm_path):
            self.skipTest(f"Test file not found: {pcm_path}")

        with open(pcm_path, "rb") as f:
            audio_data = f.read()

        queue, conn_id = self.__client.connect()
        chunk_bytes = int(16000 * 0.03 * 2)
        total_chunks = len(audio_data) // chunk_bytes
        for i in range(total_chunks):
            start = i * chunk_bytes
            end = start + chunk_bytes
            chunk = audio_data[start:end]
            await self.__client.feed(conn_id, chunk)

        silence_duration_ms = 1500
        silence_chunks_count = silence_duration_ms // 30
        silence_chunk = b'\x00' * chunk_bytes

        for _ in range(silence_chunks_count):
            await self.__client.feed(conn_id, silence_chunk)

        try:
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            self.assertIsInstance(event, StartSpeakingEvent, "Should receive StartSpeakingEvent first")
            print("Event Received: StartSpeaking")

            event = await asyncio.wait_for(queue.get(), timeout=5.0)
            self.assertIsInstance(event, StopSpeakingEvent, "Should receive StopSpeakingEvent second")
            print("Event Received: StopSpeaking")

            event = await asyncio.wait_for(queue.get(), timeout=20.0)
            self.assertIsInstance(event, TextEvent, "Should receive TextEvent last")
            print(f"Event Received: Text -> '{event.text}'")

            expected_text = "You are given an integer matrix grid and an array queries of size k."
            assert_text_similar(self, expected_text, event.text, threshold=0.8)

        except asyncio.TimeoutError:
            self.fail("Test timed out waiting for events from the pipeline.")
        except Exception as e:
            self.fail(f"An unexpected error occurred: {e}")
        finally:
            self.__client.disconnect(conn_id)


if __name__ == '__main__':
    unittest.main()
