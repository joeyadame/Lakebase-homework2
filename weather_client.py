"""
National Weather Service API client for Weather Intelligence.

The NWS API is keyless, but it requires a descriptive User-Agent header. This
client resolves user-supplied locations to NWS gridpoints, fetches active
alerts and forecast periods, and normalizes both into a common document schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

_DEFAULT_TIMEOUT = 30
_LAT_LON_RE = re.compile(
    r"^\s*(?P<lat>-?\d+(?:\.\d+)?)\s*,\s*(?P<lon>-?\d+(?:\.\d+)?)\s*$"
)
_CITY_STATE_RE = re.compile(r"^\s*(?P<city>[^,]+),\s*(?P<state>[A-Za-z]{2})\s*$")

# A small demo fallback keeps the homework examples fast and deterministic.
_COMMON_CITY_COORDS: dict[str, tuple[float, float]] = {
    "ATLANTA, GA": (33.7490, -84.3880),
    "AUSTIN, TX": (30.2672, -97.7431),
    "BOSTON, MA": (42.3601, -71.0589),
    "CHICAGO, IL": (41.8781, -87.6298),
    "DALLAS, TX": (32.7767, -96.7970),
    "DENVER, CO": (39.7392, -104.9903),
    "HOUSTON, TX": (29.7604, -95.3698),
    "LOS ANGELES, CA": (34.0522, -118.2437),
    "MIAMI, FL": (25.7617, -80.1918),
    "MINNEAPOLIS, MN": (44.9778, -93.2650),
    "NEW ORLEANS, LA": (29.9511, -90.0715),
    "NEW YORK, NY": (40.7128, -74.0060),
    "PHOENIX, AZ": (33.4484, -112.0740),
    "PORTLAND, OR": (45.5152, -122.6784),
    "SAN FRANCISCO, CA": (37.7749, -122.4194),
    "SEATTLE, WA": (47.6062, -122.3321),
    "WASHINGTON, DC": (38.9072, -77.0369),
}


class WeatherClientError(Exception):
    """Raised when a location or NWS request cannot be resolved."""


@dataclass(frozen=True)
class ResolvedLocation:
    label: str
    latitude: float
    longitude: float
    state: str | None
    office: str | None
    grid_x: int | None
    grid_y: int | None
    forecast_url: str | None
    hourly_forecast_url: str | None
    point_payload: dict[str, Any]


def _stable_hash(*parts: Any) -> str:
    raw = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return None


def _join_text(*values: Any) -> str:
    pieces: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        dedupe_key = " ".join(cleaned.lower().split())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        pieces.append(cleaned)
    return "\n\n".join(pieces)


def _load_location_overrides() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("WEATHER_LOCATION_COORDS_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WeatherClientError("WEATHER_LOCATION_COORDS_JSON is not valid JSON") from exc

    overrides: dict[str, tuple[float, float]] = {}
    for label, coords in parsed.items():
        if (
            isinstance(label, str)
            and isinstance(coords, (list, tuple))
            and len(coords) == 2
        ):
            overrides[label.strip().upper()] = (float(coords[0]), float(coords[1]))
    return overrides


class WeatherClient:
    """Thin wrapper around NWS + public geocoding services."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        user_agent: str | None = None,
    ):
        self.base_url = (base_url or os.environ.get("NWS_API_BASE_URL") or "https://api.weather.gov").rstrip("/")
        self.timeout = timeout
        self.user_agent = (
            user_agent
            or os.environ.get("NWS_USER_AGENT")
            or "weather-intelligence-lakebase-app/1.0 (student@example.com)"
        )
        self.census_base_url = (
            os.environ.get("CENSUS_GEOCODER_BASE_URL")
            or "https://geocoding.geo.census.gov"
        ).rstrip("/")
        self.nominatim_base_url = (
            os.environ.get("NOMINATIM_BASE_URL")
            or "https://nominatim.openstreetmap.org"
        ).rstrip("/")
        self._location_overrides = _load_location_overrides()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/geo+json, application/json",
                "User-Agent": self.user_agent,
            }
        )

    def get(self, path_or_url: str, params: dict[str, Any] | None = None) -> Any:
        url = self._absolute_url(path_or_url)
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def fetch_documents_for_locations(
        self,
        locations: list[str | dict[str, Any]],
        limit: int = 50,
        include_hourly: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Resolve each location and return normalized weather documents."""
        limit = max(1, int(limit))
        documents: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for raw_location in locations:
            if len(documents) >= limit:
                break
            label = self._location_label(raw_location)
            try:
                resolved = self.resolve_location(raw_location)
                remaining = limit - len(documents)
                documents.extend(self._alert_documents(resolved, remaining))

                remaining = limit - len(documents)
                if remaining > 0:
                    documents.extend(
                        self._forecast_documents(
                            resolved,
                            remaining,
                            hourly=False,
                        )
                    )

                remaining = limit - len(documents)
                if include_hourly and remaining > 0:
                    documents.extend(
                        self._forecast_documents(
                            resolved,
                            remaining,
                            hourly=True,
                        )
                    )
            except Exception as exc:
                errors.append({"location": label, "error": str(exc)})

        return documents[:limit], errors

    def resolve_location(self, value: str | dict[str, Any]) -> ResolvedLocation:
        label, latitude, longitude, state = self._resolve_lat_lon(value)
        point = self.get(f"/points/{latitude:.4f},{longitude:.4f}")
        props = point.get("properties") or {}
        relative = ((props.get("relativeLocation") or {}).get("properties") or {})
        relative_city = relative.get("city")
        relative_state = relative.get("state")
        if not state and isinstance(relative_state, str):
            state = relative_state.upper()
        if (
            not label
            and isinstance(relative_city, str)
            and isinstance(relative_state, str)
        ):
            label = f"{relative_city}, {relative_state}"

        return ResolvedLocation(
            label=label or f"{latitude:.4f},{longitude:.4f}",
            latitude=latitude,
            longitude=longitude,
            state=state,
            office=props.get("gridId"),
            grid_x=props.get("gridX"),
            grid_y=props.get("gridY"),
            forecast_url=props.get("forecast"),
            hourly_forecast_url=props.get("forecastHourly"),
            point_payload=point,
        )

    def _absolute_url(self, path_or_url: str) -> str:
        parsed = urlparse(path_or_url)
        if parsed.scheme and parsed.netloc:
            return path_or_url
        path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
        return f"{self.base_url}{path}"

    def _location_label(self, value: str | dict[str, Any]) -> str:
        if isinstance(value, dict):
            label = value.get("label") or value.get("location") or value.get("name")
            if label:
                return str(label)
            if "lat" in value and ("lon" in value or "lng" in value):
                return f"{value.get('lat')},{value.get('lon', value.get('lng'))}"
            return str(value)
        return str(value)

    def _resolve_lat_lon(
        self,
        value: str | dict[str, Any],
    ) -> tuple[str, float, float, str | None]:
        if isinstance(value, dict):
            label = self._location_label(value)
            lat = value.get("lat") or value.get("latitude")
            lon = value.get("lon") or value.get("lng") or value.get("longitude")
            state = value.get("state")
            if lat is not None and lon is not None:
                return label, float(lat), float(lon), str(state).upper() if state else None
            raise WeatherClientError(f"Location object is missing lat/lon: {value!r}")

        label = value.strip()
        if not label:
            raise WeatherClientError("Location cannot be blank")

        lat_lon = _LAT_LON_RE.match(label)
        if lat_lon:
            return (
                label,
                float(lat_lon.group("lat")),
                float(lat_lon.group("lon")),
                None,
            )

        city_state = _CITY_STATE_RE.match(label)
        state = city_state.group("state").upper() if city_state else None
        key = label.upper()
        if key in self._location_overrides:
            lat, lon = self._location_overrides[key]
            return label, lat, lon, state
        if key in _COMMON_CITY_COORDS:
            lat, lon = _COMMON_CITY_COORDS[key]
            return label, lat, lon, state

        lat_lon_result = self._geocode_with_census(label) or self._geocode_with_nominatim(label)
        if lat_lon_result is None:
            raise WeatherClientError(
                f"Could not geocode {label!r}; pass a 'lat,lon' string or add it to WEATHER_LOCATION_COORDS_JSON"
            )
        lat, lon = lat_lon_result
        return label, lat, lon, state

    def _geocode_with_census(self, label: str) -> tuple[float, float] | None:
        try:
            resp = self._session.get(
                f"{self.census_base_url}/geocoder/locations/onelineaddress",
                params={
                    "address": label,
                    "benchmark": "Public_AR_Current",
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            matches = (resp.json().get("result") or {}).get("addressMatches") or []
            if not matches:
                return None
            coords = matches[0].get("coordinates") or {}
            lon = coords.get("x")
            lat = coords.get("y")
            if lat is None or lon is None:
                return None
            return float(lat), float(lon)
        except Exception:
            return None

    def _geocode_with_nominatim(self, label: str) -> tuple[float, float] | None:
        if os.environ.get("WEATHER_DISABLE_NOMINATIM", "").lower() == "true":
            return None
        try:
            resp = self._session.get(
                f"{self.nominatim_base_url}/search",
                params={
                    "q": label,
                    "format": "json",
                    "limit": 1,
                    "countrycodes": "us",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            matches = resp.json()
            if not matches:
                return None
            return float(matches[0]["lat"]), float(matches[0]["lon"])
        except Exception:
            return None

    def _alert_documents(
        self,
        resolved: ResolvedLocation,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []

        data = self.get(
            "/alerts/active",
            params={"point": f"{resolved.latitude:.4f},{resolved.longitude:.4f}"},
        )
        features = data.get("features") or []
        documents: list[dict[str, Any]] = []
        for feature in features[:limit]:
            props = feature.get("properties") or {}
            event = _first_text(props.get("event"), props.get("headline")) or "Weather Alert"
            headline = _first_text(props.get("headline"), event) or event
            narrative = _join_text(
                headline,
                props.get("description"),
                props.get("instruction"),
            )
            if not narrative:
                continue
            alert_id = props.get("id") or feature.get("id") or _stable_hash(
                resolved.label,
                event,
                props.get("sent"),
                narrative,
            )
            documents.append(
                {
                    "id": f"nws:alert:{alert_id}",
                    "location": resolved.label,
                    "latitude": resolved.latitude,
                    "longitude": resolved.longitude,
                    "office": resolved.office,
                    "grid_x": resolved.grid_x,
                    "grid_y": resolved.grid_y,
                    "source_type": "alert",
                    "headline": headline,
                    "event": event,
                    "narrative_text": narrative,
                    "issued_at": props.get("sent") or props.get("onset"),
                    "effective_at": props.get("effective") or props.get("onset"),
                    "expires_at": props.get("expires") or props.get("ends"),
                    "payload": {
                        "source": "api.weather.gov",
                        "endpoint": "/alerts/active",
                        "resolved_location": _resolved_payload(resolved),
                        "raw": feature,
                    },
                }
            )
        return documents

    def _forecast_documents(
        self,
        resolved: ResolvedLocation,
        limit: int,
        hourly: bool = False,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        forecast_url = resolved.hourly_forecast_url if hourly else resolved.forecast_url
        if not forecast_url:
            return []

        data = self.get(forecast_url)
        props = data.get("properties") or {}
        periods = props.get("periods") or []
        forecast_kind = "hourly_forecast" if hourly else "forecast"
        documents: list[dict[str, Any]] = []

        for period in periods[:limit]:
            start = period.get("startTime")
            end = period.get("endTime")
            name = _first_text(period.get("name")) or "Forecast"
            short = _first_text(period.get("shortForecast"))
            detailed = _first_text(period.get("detailedForecast"), short)
            headline = _join_text(name, short).replace("\n\n", ": ", 1)

            weather_bits = [
                detailed,
                _weather_measurement_text(period),
            ]
            narrative = _join_text(headline, *weather_bits)
            if not narrative:
                continue

            doc_id = _stable_hash(
                forecast_kind,
                resolved.office,
                resolved.grid_x,
                resolved.grid_y,
                start,
                period.get("number"),
            )
            documents.append(
                {
                    "id": f"nws:{forecast_kind}:{doc_id}",
                    "location": resolved.label,
                    "latitude": resolved.latitude,
                    "longitude": resolved.longitude,
                    "office": resolved.office,
                    "grid_x": resolved.grid_x,
                    "grid_y": resolved.grid_y,
                    "source_type": "forecast",
                    "headline": headline,
                    "event": short or name,
                    "narrative_text": narrative,
                    "issued_at": props.get("generatedAt") or props.get("updateTime"),
                    "effective_at": start,
                    "expires_at": end,
                    "payload": {
                        "source": "api.weather.gov",
                        "endpoint": forecast_url,
                        "forecast_kind": forecast_kind,
                        "generated_at": props.get("generatedAt"),
                        "updated_at": props.get("updateTime"),
                        "resolved_location": _resolved_payload(resolved),
                        "raw": period,
                    },
                }
            )
        return documents


def _weather_measurement_text(period: dict[str, Any]) -> str:
    pieces: list[str] = []
    temp = period.get("temperature")
    temp_unit = period.get("temperatureUnit")
    if temp is not None and temp_unit:
        pieces.append(f"Temperature {temp} {temp_unit}.")
    wind_speed = period.get("windSpeed")
    wind_direction = period.get("windDirection")
    if wind_speed or wind_direction:
        pieces.append(
            "Wind "
            + " ".join(str(part) for part in (wind_direction, wind_speed) if part)
            + "."
        )
    precipitation = period.get("probabilityOfPrecipitation") or {}
    precip_value = precipitation.get("value")
    if precip_value is not None:
        pieces.append(f"Precipitation chance {precip_value} percent.")
    return " ".join(pieces)


def _resolved_payload(resolved: ResolvedLocation) -> dict[str, Any]:
    return {
        "label": resolved.label,
        "latitude": resolved.latitude,
        "longitude": resolved.longitude,
        "state": resolved.state,
        "office": resolved.office,
        "grid_x": resolved.grid_x,
        "grid_y": resolved.grid_y,
        "forecast_url": resolved.forecast_url,
        "hourly_forecast_url": resolved.hourly_forecast_url,
    }
