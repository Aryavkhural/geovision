from datetime import timedelta
from io import BytesIO

import numpy as np
import requests
import tifffile


TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


def make_bbox(latitude, longitude, width_km):
    half_lat = (width_km / 2) / 111.32

    half_lon = (width_km / 2) / (
        111.32 * max(np.cos(np.radians(latitude)), 0.1)
    )

    return [
        longitude - half_lon,
        latitude - half_lat,
        longitude + half_lon,
        latitude + half_lat,
    ]


def get_access_token(client_id, client_secret):
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )

    response.raise_for_status()

    token_data = response.json()
    return token_data["access_token"]


def get_sentinel2_live_features(
    latitude,
    longitude,
    width_km,
    end_date,
    client_id,
    client_secret,
):
    bbox = make_bbox(latitude, longitude, width_km)
    start_date = end_date - timedelta(days=60)

    width_pixels = max(32, int((width_km * 1000) / 10))
    height_pixels = max(32, int((width_km * 1000) / 10))

    evalscript = """
    //VERSION=3

    function setup() {
      return {
        input: [{
          bands: ["B02", "B03", "B04", "B08", "B11", "B12", "dataMask"],
          units: "REFLECTANCE"
        }],
        output: {
          bands: 10,
          sampleType: "FLOAT32"
        }
      };
    }

    function evaluatePixel(sample) {
      let eps = 0.000001;

      let ndvi = (sample.B08 - sample.B04) /
                 (sample.B08 + sample.B04 + eps);

      let ndwi = (sample.B03 - sample.B08) /
                 (sample.B03 + sample.B08 + eps);

      let ndbi = (sample.B11 - sample.B08) /
                 (sample.B11 + sample.B08 + eps);

      return [
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B08,
        sample.B11,
        sample.B12,
        ndvi,
        ndwi,
        ndbi,
        sample.dataMask
      ];
    }
    """

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
                },
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{start_date.isoformat()}T00:00:00Z",
                            "to": f"{end_date.isoformat()}T23:59:59Z",
                        },
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": width_pixels,
            "height": height_pixels,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff"
                    },
                }
            ],
        },
        "evalscript": evalscript,
    }

    access_token = get_access_token(client_id, client_secret)

    response = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    image = tifffile.imread(
        BytesIO(response.content)
    ).astype(np.float32)

    features = image[:, :, :9]
    valid_mask = image[:, :, 9] > 0

    return features, valid_mask, bbox, start_date