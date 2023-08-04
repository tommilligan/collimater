#!/usr/bin/env python

import argparse
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Callable, Iterable, List, Optional, Protocol

import recurring_ical_events
from dotenv import load_dotenv
from icalendar import Calendar, Event
from luma.core.error import DeviceNotFoundError
from luma.core.interface.serial import noop, spi
from luma.core.virtual import sevensegment
from luma.led_matrix.device import max7219
from requests import Session
from requests.adapters import HTTPAdapter, Retry

_log = logging.getLogger(__file__)


def poll_until_shutdown(
    callable: Callable[[], None],
    poll_interval: float,
    shutdown_event: Event,
) -> None:
    _log.info("Polling every %.1f seconds...", poll_interval)
    while not shutdown_event.is_set():
        start_time = time.monotonic()
        callable()
        time_taken = time.monotonic() - start_time

        delay = poll_interval - time_taken
        if delay > 0.0:
            # Wait for the remainder of the poll interval
            # Unless we get shutdown, then return immediately.
            shutdown_event.wait(delay)


class ConfigurationError(Exception):
    """Raised when there is a misconfiguration."""


class Printer(Protocol):
    def print(self, value: str) -> None:
        ...


class PrinterCli:
    def print(self, value: str) -> Calendar:
        print(value)


class PrinterZeroseg:
    _seg: sevensegment

    def __init__(self):
        serial = spi(port=0, device=0, gpio=noop())
        device = max7219(serial, cascaded=1)
        seg = sevensegment(device)
        self._seg = seg

    def print(self, value: str) -> Calendar:
        self._seg.text = value


class CalendarFetcher(Protocol):
    def fetch(self) -> Calendar:
        ...


class CalendarFetcherRemote:
    _session: Session
    _ics_url: str

    def __init__(self, session: Session, ics_url: str) -> None:
        self._session = session
        self._ics_url = ics_url

    def fetch(self) -> Calendar:
        ics = self._session.get(self._ics_url).text
        calendar = Calendar.from_ical(ics)
        return calendar


class CalendarFetcherLocal:
    _path: str

    def __init__(self, path: str) -> None:
        self._path = path

    def fetch(self) -> Calendar:
        with open(self._path, "r") as fh:
            ics = fh.read()
        calendar = Calendar.from_ical(ics)
        return calendar


class Freebusy(Enum):
    FREE = "FREE"
    BUSY = "BUSY"

    @staticmethod
    def try_from(other: str) -> Optional["Freebusy"]:
        try:
            return Freebusy(other)
        except ValueError:
            return None


@dataclass(frozen=True)
class Mark:
    """A specific datetime, with associated data.

    A mark for the user to hit accurately.
    """

    at: datetime
    description: str
    freebusy: Freebusy

    @staticmethod
    def from_event(event: Event) -> Optional["Mark"]:
        at = event["DTSTART"].dt

        # Skip all day events
        if type(at) == date:
            return None

        description = event["SUMMARY"]
        freebusy = (
            Freebusy.try_from(event.get("X-MICROSOFT-CDO-BUSYSTATUS", ""))
            or Freebusy.BUSY
        )
        return Mark(at=at, description=description, freebusy=freebusy)


def extract_marks(events: Iterable[Event]) -> List[Mark]:
    marks = []
    for event in events:
        mark = Mark.from_event(event)
        if mark is not None:
            marks.append(mark)

    marks.sort(key=lambda mark: mark.at)
    return marks


class RelevantReason(Enum):
    IMMINENT = "imminent"
    RECENT = "recent"
    SCHEDULED = "scheduled"

    @staticmethod
    def try_from(other: str) -> Optional["Freebusy"]:
        try:
            return Freebusy(other)
        except ValueError:
            return None


@dataclass(frozen=True)
class RelevantMark:
    mark: Mark
    reason: RelevantReason


@dataclass(frozen=True)
class RelevantCriteria:
    imminent: float
    recent: float
    scheduled: float


def find_relevant_mark(
    marks: List[Mark], criteria: RelevantCriteria, now: datetime
) -> Optional[RelevantMark]:
    """Given a list of sorted marks, finds the most relevant one to the given
    datetime.

    This is:

    - If there is an event close in the future (< imminent event threshold)
    - Or, if there is an event close in the past (< recent event threshold)
    - Or, if there is an event scheduled in future (< scheduled event threshold)

    The mark is returned, plus the classification of which kind of event it is.
    """
    past = None
    future = None

    for mark in marks:
        if mark.freebusy is Freebusy.FREE:
            continue

        if mark.at < now:
            past = mark
        else:
            future = mark
            break

    if future is not None and (future.at - now).seconds <= criteria.imminent:
        return RelevantMark(mark=future, reason=RelevantReason.IMMINENT)

    if past is not None and (now - past.at).seconds <= criteria.recent:
        return RelevantMark(mark=past, reason=RelevantReason.RECENT)

    if future is not None and (future.at - now).seconds <= criteria.scheduled:
        return RelevantMark(mark=future, reason=RelevantReason.SCHEDULED)

    return None


class LoadingIndicator:
    def __init__(self) -> None:
        self._index = 0
        self._frames = [
            ".       ",
            " .      ",
            "  .     ",
            "   .    ",
            "    .   ",
            "     .  ",
            "      . ",
            "       .",
        ]

    def tick(self) -> None:
        self._index = (self._index + 1) % len(self._frames)

    def display(self) -> str:
        return self._frames[self._index]


class RelevantMarkExtractor:
    _poll_interval: float
    _calendar_queue: queue.Queue
    _calender: Optional[Calendar]
    _relevant_mark_queue: queue.Queue

    def __init__(
        self,
        poll_interval: float,
        shutdown_event: threading.Event,
        calendar_queue: queue.Queue,
        relevant_mark_queue: queue.Queue,
    ) -> None:
        self._poll_interval = poll_interval
        self._shutdown_event = shutdown_event
        self._calendar_queue = calendar_queue
        self._relevant_mark_queue = relevant_mark_queue

        # state
        self._calendar = None

    def run(self) -> None:
        poll_until_shutdown(
            self.poll,
            self._poll_interval,
            self._shutdown_event,
        )

    def poll(self) -> None:
        try:
            self._calendar = self._calendar_queue.get(timeout=self._poll_interval)
        except queue.Empty:
            pass

        if self._calendar is None:
            return

        _log.info("Extracting relevant mark")
        now = datetime.now(tz=timezone.utc)
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=2)

        events = recurring_ical_events.of(self._calendar).between(start_date, end_date)
        marks = extract_marks(events)

        relevant_mark = find_relevant_mark(
            marks,
            criteria=RelevantCriteria(
                # FIXME
                imminent=5.0 * 60.0 * 100.0,
                recent=10.0 * 60.0 * 100.0,
                scheduled=10.0 * 60.0,
            ),
            now=now,
        )
        _log.info(f"Extracted relevant mark: {relevant_mark}")
        try:
            self._relevant_mark_queue.put(relevant_mark)
        except queue.Full:
            pass


class DisplayManager:
    _poll_interval: float
    _relevant_mark_queue: queue.Queue
    _printer: Printer

    _relevant_mark: Optional[RelevantMark]
    _loading_indicator: LoadingIndicator

    def __init__(
        self,
        poll_interval: float,
        shutdown_event: threading.Event,
        relevant_mark_queue: queue.Queue,
        printer: Printer,
    ) -> None:
        self._poll_interval = poll_interval
        self._shutdown_event = shutdown_event
        self._relevant_mark_queue = relevant_mark_queue
        self._printer = printer

        # state
        self._relevant_mark = None
        self._loading_indicator = LoadingIndicator()

    def run(self) -> None:
        poll_until_shutdown(
            self.poll,
            self._poll_interval,
            self._shutdown_event,
        )

    def poll(self) -> None:
        try:
            self._relevant_mark = self._relevant_mark_queue.get(block=False)
        except queue.Empty:
            pass

        if self._relevant_mark is None:
            self._printer.print(self._loading_indicator.display())
            self._loading_indicator.tick()
            return

        now = datetime.now(tz=timezone.utc)
        self._printer.print(format_timedelta(now - self._relevant_mark.mark.at))


class Collimater:
    _calendar_fetcher: CalendarFetcher
    _calendar_fetch_interval: float
    _relevant_mark_interval: float
    _printer: Printer
    _print_interval: float
    _shutdown_event: threading.Event

    def __init__(
        self,
        calendar_fetcher: CalendarFetcher,
        calendar_fetch_interval: float,
        relevant_mark_interval: float,
        printer: Printer,
        print_interval: float,
        shutdown_event: threading.Event,
    ) -> None:
        self._calendar_fetcher = calendar_fetcher
        self._calendar_fetch_interval = calendar_fetch_interval
        self._relevant_mark_interval = relevant_mark_interval
        self._printer = printer
        self._print_interval = print_interval
        self._shutdown_event = shutdown_event

        # state
        self._calendar = None

    def run(self) -> None:
        calendar_queue = queue.Queue(1)
        relevant_mark_queue = queue.Queue(1)

        def calendar_fetcher_loop() -> None:
            _log.info("Fetching calendar")
            calendar = self._calendar_fetcher.fetch()
            try:
                calendar_queue.put(calendar)
            except queue.Full:
                pass
            _log.info("Fetched calendar")

        def calender_fetcher_run() -> None:
            poll_until_shutdown(
                calendar_fetcher_loop,
                self._calendar_fetch_interval,
                self._shutdown_event,
            )

        threading.Thread(
            target=calender_fetcher_run,
            args=(),
            name="calendar_fetcher",
            daemon=True,
        ).start()

        relevant_mark_extractor = RelevantMarkExtractor(
            poll_interval=self._relevant_mark_interval,
            shutdown_event=self._shutdown_event,
            calendar_queue=calendar_queue,
            relevant_mark_queue=relevant_mark_queue,
        )

        threading.Thread(
            target=relevant_mark_extractor.run,
            args=(),
            name="relevant_mark_extractor",
            daemon=True,
        ).start()

        display_manager = DisplayManager(
            poll_interval=self._print_interval,
            shutdown_event=self._shutdown_event,
            relevant_mark_queue=relevant_mark_queue,
            printer=self._printer,
        )
        display_manager_thread = threading.Thread(
            target=display_manager.run,
            args=(),
            name="display_manager",
            daemon=True,
        )
        display_manager_thread.start()
        display_manager_thread.join()


def left_pad_excluding_periods(
    value: str, target_length: str, padding_character: str
) -> str:
    padding = padding_character * (target_length - (len(value) - value.count(".")))
    return padding + value


def format_timedelta(delta: timedelta) -> str:
    total_seconds = delta.total_seconds()
    sign = "-" if total_seconds < 0 else " "
    remainder = abs(total_seconds)
    # hours
    hours = int(remainder // 3600)
    remainder = remainder - (hours * 3600)
    # minutes
    minutes = int(remainder // 60)
    seconds = int(remainder - (minutes * 60))

    seconds_format = "02" if minutes > 0 else ""
    minutes_format = "02" if hours > 0 else ""

    display = f"{seconds:{seconds_format}}"
    if minutes > 0:
        display = f"{minutes:{minutes_format}}.{display}"
    if hours > 0:
        display = f"{hours}.{display}"

    display = f"{sign}{display}"
    return left_pad_excluding_periods(display, 8, " ")


def run(args: argparse.Namespace) -> None:
    session = Session()

    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])

    session.mount("https://", HTTPAdapter(max_retries=retries))

    if args.calendar_file is not None:
        calendar_fetcher = CalendarFetcherLocal(path=args.calendar_file)
    else:
        ics_url = os.getenv("COLLIMATER_ICS_URL")
        if ics_url is None:
            raise ConfigurationError(
                "COLLIMATER_ICS_URL must be set to a syncable calendar"
            )

        calendar_fetcher = CalendarFetcherRemote(session=session, ics_url=ics_url)

    try:
        printer = PrinterZeroseg()
    except DeviceNotFoundError:
        printer = PrinterCli()

    shutdown_event = threading.Event()
    Collimater(
        calendar_fetcher=calendar_fetcher,
        calendar_fetch_interval=args.calendar_fetch_interval,
        relevant_mark_interval=args.relevant_mark_interval,
        printer=printer,
        print_interval=args.print_interval,
        shutdown_event=shutdown_event,
    ).run()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="collimater - keep yourself on track")
    parser.add_argument(
        "--calendar-file",
        type=str,
        help="The local calendar file to use. Defaults to remote url",
    )
    parser.add_argument(
        "--calendar-fetch-interval",
        type=float,
        default=600.0,
        help="Interval (secs) to poll for calendar changes",
    )
    parser.add_argument(
        "--relevant-mark-interval",
        type=float,
        default=60.0,
        help="Interval (secs) to recompute relevant mark",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.1,
        help="The interval to refresh the display output",
    )
    return parser


def setup_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s|%(name)s|%(threadName)s|%(levelname)s|%(message)s"
        )
    )
    root_logger.handlers = [stream_handler]


def main() -> None:
    setup_logging()
    load_dotenv()

    parser = make_parser()
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
