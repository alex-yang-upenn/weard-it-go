class UniformFrameSampler:
    def __init__(self, frames_per_second: float):
        self.sample_interval_ms = 1000 / frames_per_second
        self.next_sample_at_ms = 0.0

    def should_sample(self, received_at_ms: int) -> bool:
        if received_at_ms < self.next_sample_at_ms:
            return False
        self.next_sample_at_ms += self.sample_interval_ms
        if self.next_sample_at_ms <= received_at_ms:
            self.next_sample_at_ms = received_at_ms + self.sample_interval_ms
        return True
