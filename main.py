#!/usr/bin/env python

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, List, Optional, Protocol

import recurring_ical_events
from dotenv import load_dotenv
from icalendar import Calendar, Event
from requests import Session
from requests.adapters import HTTPAdapter, Retry

_log = logging.getLogger(__file__)


class ConfigurationError(Exception):
    """Raised when there is a misconfiguration."""


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
    freebusy: Optional[Freebusy]

    @staticmethod
    def from_event(event: Event) -> Optional["Mark"]:
        at = event["DTSTART"].dt

        # Skip all day events
        if type(at) == date:
            return None

        description = event["SUMMARY"]
        freebusy = Freebusy.try_from(event.get("X-MICROSOFT-CDO-BUSYSTATUS", ""))
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


class Printer:
    _ics_url: str

    def __init__(self, session: Session, ics_url: str) -> None:
        self._session = session
        self._ics_url = ics_url

    def fetch(self) -> Calendar:
        ics = self._session.get(self._ics_url).text
        calendar = Calendar.from_ical(ics)
        return calendar


class Collimater:
    _session: Session

    def __init__(self, calendar_fetcher: CalendarFetcher) -> None:
        self._calendar_fetcher = calendar_fetcher

    def run(self) -> None:
        self.poll()

    def poll(self) -> None:
        calendar = self._calendar_fetcher.fetch()
        now = datetime.now()
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=2)

        events = recurring_ical_events.of(calendar).between(start_date, end_date)
        marks = extract_marks(events)

        relevant_mark = find_relevant_mark(
            marks,
            criteria=RelevantCriteria(
                # FIXME
                imminent=5.0 * 60.0 * 100.0,
                recent=10.0 * 60.0,
                scheduled=10.0 * 60.0,
            ),
            now=datetime.now(tz=timezone.utc),
        )
        _log.info(f"Relevant mark: {relevant_mark}")


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

    Collimater(calendar_fetcher=calendar_fetcher).run()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="collimater - keep yourself on track")
    parser.add_argument(
        "--calendar-file",
        type=str,
        help="The local calendar file to use. Defaults to remote url",
    )
    parser.add_argument(
        "--calendar-poll-interval",
        type=float,
        default=600.0,
        help="Interval (secs) to poll for calendar changes",
    )
    return parser


def setup_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s|%(name)s|%(levelname)s|%(message)s")
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
