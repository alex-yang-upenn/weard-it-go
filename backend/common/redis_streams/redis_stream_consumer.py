import redis.asyncio as redis
from redis.exceptions import ResponseError


class RedisStreamConsumer:
    def __init__(self, redis_client: redis.Redis, stream_name: str, group_name: str, consumer_name: str):
        self.redis_client = redis_client
        self.stream_name = stream_name
        self.group_name = group_name
        self.consumer_name = consumer_name
        self.unacknowledged_messages_from_last_run_drained = False

    async def create_group_if_missing(self) -> None:
        try:
            await self.redis_client.xgroup_create(self.stream_name, self.group_name, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def read_batch(self, max_messages: int, block_milliseconds: int = 1000) -> list[tuple[bytes, dict]]:
        if not self.unacknowledged_messages_from_last_run_drained:
            unacknowledged_messages = await self.read_from_stream_position("0", max_messages, block_milliseconds)
            if unacknowledged_messages:
                return unacknowledged_messages
            self.unacknowledged_messages_from_last_run_drained = True
        return await self.read_from_stream_position(">", max_messages, block_milliseconds)

    async def read_from_stream_position(
        self, stream_position: str, max_messages: int, block_milliseconds: int
    ) -> list[tuple[bytes, dict]]:
        response = await self.redis_client.xreadgroup(
            self.group_name,
            self.consumer_name,
            {self.stream_name: stream_position},
            count=max_messages,
            block=block_milliseconds,
        )
        if not response:
            return []
        return response[0][1]

    async def acknowledge(self, message_ids: list[bytes]) -> None:
        if message_ids:
            await self.redis_client.xack(self.stream_name, self.group_name, *message_ids)
