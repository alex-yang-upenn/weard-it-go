import asyncio
import json
import time

from livekit import rtc

UNIVERSITY_OF_PENNSYLVANIA_LATITUDE = 39.9522
UNIVERSITY_OF_PENNSYLVANIA_LONGITUDE = -75.1932
LATITUDE_DRIFT_PER_SECOND = 0.00001


class FakeGpsPublisher:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.latitude = UNIVERSITY_OF_PENNSYLVANIA_LATITUDE

    async def publish_every_second(self) -> None:
        while True:
            self.latitude += LATITUDE_DRIFT_PER_SECOND
            gps_sample = {
                "latitude": self.latitude,
                "longitude": UNIVERSITY_OF_PENNSYLVANIA_LONGITUDE,
                "accuracy_m": 5,
                "phone_time_ms": int(time.time() * 1000),
            }
            await self.room.local_participant.publish_data(json.dumps(gps_sample), reliable=True, topic="telemetry")
            await asyncio.sleep(1)
