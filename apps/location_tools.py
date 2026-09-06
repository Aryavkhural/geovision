from geopy.geocoders import Nominatim


def reverse_geocode_india(latitude, longitude):
    try:
        geocoder = Nominatim(
            user_agent="geovision-india-lulc-prototype"
        )

        result = geocoder.reverse(
            (latitude, longitude),
            language="en",
            zoom=10,
            exactly_one=True,
            timeout=10,
        )

        if result is None:
            return f"Custom area: {latitude:.4f}, {longitude:.4f}"

        address = result.raw.get("address", {})

        place = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state_district")
            or "Selected area"
        )

        state = address.get("state", "")
        country = address.get("country", "")

        return ", ".join(
            part for part in [place, state, country] if part
        )

    except Exception:
        return f"Custom area: {latitude:.4f}, {longitude:.4f}"