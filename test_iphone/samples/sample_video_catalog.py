from dataclasses import dataclass


@dataclass(frozen=True)
class MixkitClipSegment:
    mixkit_page_slug: str
    start_seconds: float
    duration_seconds: float

    @property
    def mixkit_video_id(self) -> str:
        return self.mixkit_page_slug.rsplit("-", 1)[1]

    @property
    def page_url(self) -> str:
        return f"https://mixkit.co/free-stock-video/{self.mixkit_page_slug}/"

    @property
    def download_url(self) -> str:
        return f"https://assets.mixkit.co/videos/{self.mixkit_video_id}/{self.mixkit_video_id}-720.mp4"


@dataclass(frozen=True)
class SampleVideo:
    name: str
    segments: tuple[MixkitClipSegment, ...]


SAMPLE_VIDEOS = (
    SampleVideo("phone_on_laptop", (
        MixkitClipSegment("black-phone-on-laptop-247", 0, 12),
        MixkitClipSegment("minimalist-room-with-gray-sofa-3110", 0, 10),
    )),
    SampleVideo("watch_at_keyboard", (
        MixkitClipSegment("business-person-working-with-devices-1814", 0, 6),
        MixkitClipSegment("narrow-and-old-alley-in-venice-4600", 0, 10),
    )),
    SampleVideo("laptop_by_the_window", (
        MixkitClipSegment("a-woman-working-on-her-laptop-at-an-office-1800", 0, 10),
        MixkitClipSegment("interior-of-a-department-room-3112", 0, 10),
    )),
    SampleVideo("first_person_laptop_with_coffee", (
        MixkitClipSegment("on-a-round-table-the-hands-of-a-young-man-52076", 0, 7.5),
        MixkitClipSegment("a-woman-walking-through-a-vintage-house-corridor-41512", 0, 10),
    )),
)


def find_sample_video(name: str) -> SampleVideo | None:
    return next((sample for sample in SAMPLE_VIDEOS if sample.name == name), None)
