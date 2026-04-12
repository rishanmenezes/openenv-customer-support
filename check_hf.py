import httpx
try:
    resp = httpx.get("https://huggingface.co/api/spaces/rizzshan/openenv-customer-support")
    print("Status:", resp.json().get("runtime", {}).get("stage"))
except Exception as e:
    print("Error:", e)
