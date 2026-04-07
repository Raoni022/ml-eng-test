from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_detect_returns_expected_fields():
    with open("tests/fixtures/sample.png", "rb") as f:
        response = client.post(
            "/detect",
            files={"file": ("sample.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()

    assert "annotated_image_base64" in data
    assert "wall_segment_count" in data
    assert "room_count" in data
    assert "room_areas_px" in data
    assert "image_width" in data
    assert "image_height" in data
    assert "processing_time_ms" in data
