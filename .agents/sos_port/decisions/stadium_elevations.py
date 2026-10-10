"""Source the coordinates and ground elevation of venues, for the stadium table's entries.

Run from the root of an ``nfl-predictor`` checkout (network access needed, nothing is written):

    .venv/bin/python .agents/sos_port/decisions/stadium_elevations.py

For each venue in ``VENUES`` it reads the venue's coordinates from its English Wikipedia article
(the MediaWiki API, ``prop=coordinates``, with the revision id it read) and asks two terrain
models for the ground elevation: the USGS Elevation Point Query Service (3DEP, a bare-earth model
of about 10 m resolution; United States only) at the coordinates, and the Open-Meteo elevation
API (Copernicus GLO-90) at the coordinates and at eight points on a ring ``RING_METRES`` around
them. Copernicus GLO-90 is a surface model, so a stadium's stands and roof raise its value at the
coordinates; the ring's median reads the ground around the site. It prints one line per venue, in
feet, rounded to the foot. With ``--existing`` it does the same for the table's entries named in
``EXISTING``, for comparison with the values the table already holds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.parse
import urllib.request
from typing import Any

from nfl_predictor import constants

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USGS_EPQS = "https://epqs.nationalmap.gov/v1/json"
OPEN_METEO = "https://api.open-meteo.com/v1/elevation"
USER_AGENT = "nfl-predictor stadium sourcing (personal research script)"
TIMEOUT_SECONDS = 60
FEET_PER_METRE = 3.28084
METRES_PER_DEGREE = 111_320.0
RING_METRES = 300.0
RING_POINTS = 8
# Venue id -> (Wikipedia article title, in the United States).
VENUES: dict[str, tuple[str, bool]] = {
    "BOS99": ("Foxboro Stadium", True),
    "CHI99": ("Memorial Stadium (Champaign)", True),
    "CIN99": ("Riverfront Stadium", True),
    "DEN99": ("Mile High Stadium", True),
    "DET99": ("Pontiac Silverdome", True),
    "PHI99": ("Veterans Stadium", True),
    "PIT99": ("Three Rivers Stadium", True),
    "SEA98": ("Kingdome", True),
    "SEA99": ("Husky Stadium", True),
    "MAD01": ("Santiago Bernabéu Stadium", False),
    "MEL00": ("Melbourne Cricket Ground", False),
    "MUN01": ("Allianz Arena", False),
    "PAR00": ("Stade de France", False),
    "RIO00": ("Maracanã Stadium", False),
}
# Entries already in the table whose sites sit next to, or are, a new venue's.
EXISTING: dict[str, tuple[str, bool]] = {
    "DEN00": ("Empower Field at Mile High", True),
    "GER00": ("Allianz Arena", False),
    "NOR00": ("Caesars Superdome", True),
    "SEA00": ("Lumen Field", True),
    "PIT00": ("Acrisure Stadium", True),
    "PHI00": ("Lincoln Financial Field", True),
    "CIN00": ("Paycor Stadium", True),
    "BOS00": ("Gillette Stadium", True),
    "DET00": ("Ford Field", True),
    "CHI98": ("Soldier Field", True),
}


def _get_json(url: str, params: dict[str, str]) -> Any:
    """Return the JSON a GET request answers, after a pause that keeps the request rate low."""
    time.sleep(1.0)
    request = urllib.request.Request(
        f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def _coordinates(titles: list[str]) -> dict[str, tuple[float, float, int, str]]:
    """Return each article's latitude, longitude, revision id and resolved title, in one query."""
    answer = _get_json(
        WIKIPEDIA_API,
        {
            "action": "query",
            "prop": "coordinates|revisions",
            "colimit": "max",
            "rvprop": "ids",
            "titles": "|".join(titles),
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        },
    )
    query = answer["query"]
    renamed = {item["from"]: item["to"] for item in query.get("normalized", [])}
    renamed |= {item["from"]: item["to"] for item in query.get("redirects", [])}
    pages = {page["title"]: page for page in query["pages"]}
    found: dict[str, tuple[float, float, int, str]] = {}
    for title in titles:
        resolved = renamed.get(renamed.get(title, title), renamed.get(title, title))
        page = pages[resolved]
        point = page["coordinates"][0]
        found[title] = (
            float(point["lat"]),
            float(point["lon"]),
            int(page["revisions"][0]["revid"]),
            resolved,
        )
    return found


def _usgs_feet(latitude: float, longitude: float) -> float | None:
    """Return the USGS 3DEP ground elevation in feet, or None when it has no value there."""
    answer = _get_json(
        USGS_EPQS,
        {"x": str(longitude), "y": str(latitude), "wkid": "4326", "units": "Feet"},
    )
    value = answer.get("value")
    return (
        float(value) if isinstance(value, (int, float, str)) and str(value) != "-1000000" else None
    )


def _open_meteo_feet(points: list[tuple[float, float]]) -> list[float]:
    """Return the Copernicus GLO-90 ground elevation in feet of each point, in one query."""
    answer = _get_json(
        OPEN_METEO,
        {
            "latitude": ",".join(f"{latitude:.5f}" for latitude, _ in points),
            "longitude": ",".join(f"{longitude:.5f}" for _, longitude in points),
        },
    )
    return [float(metres) * FEET_PER_METRE for metres in answer["elevation"]]


def _ring(latitude: float, longitude: float) -> list[tuple[float, float]]:
    """Return ``RING_POINTS`` points evenly spaced on a circle ``RING_METRES`` around a point."""
    lat_step = RING_METRES / METRES_PER_DEGREE
    lon_step = RING_METRES / (METRES_PER_DEGREE * math.cos(math.radians(latitude)))
    angles = [2 * math.pi * index / RING_POINTS for index in range(RING_POINTS)]
    return [
        (latitude + lat_step * math.sin(angle), longitude + lon_step * math.cos(angle))
        for angle in angles
    ]


def _report(venues: dict[str, tuple[str, bool]]) -> None:
    """Print the coordinates and elevations of each venue."""
    located = _coordinates([title for title, _ in venues.values()])
    for venue, (title, in_us) in venues.items():
        latitude, longitude, revision, resolved = located[title]
        center, *ring = _open_meteo_feet([(latitude, longitude), *_ring(latitude, longitude)])
        usgs = _usgs_feet(latitude, longitude) if in_us else None
        table = constants.STADIUMS.get(venue, {}).get("elevation_ft")
        sys.stdout.write(
            f"{venue} | {resolved} (revision {revision}) | {latitude:.5f}, {longitude:.5f} | "
            f"USGS 3DEP {'-' if usgs is None else round(usgs)} ft | "
            f"Copernicus GLO-90 {round(center)} ft, ring median {round(statistics.median(ring))} "
            f"ft | table {table}\n"
        )


def main() -> int:
    """Print the venue sources and return the exit status."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--existing", action="store_true", help="Also source EXISTING entries.")
    arguments = parser.parse_args()
    sys.stdout.write("venue | Wikipedia article | latitude, longitude | elevations | table\n")
    _report(VENUES)
    if arguments.existing:
        sys.stdout.write("\nEntries already in the table, for comparison:\n")
        _report(EXISTING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
