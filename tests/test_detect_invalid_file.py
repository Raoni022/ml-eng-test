from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_detect_rejects_invalid_file_type():
    response = client.post(
        "/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415
