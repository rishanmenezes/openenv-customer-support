import httpx
import json

def test_endpoint(url):
    print(f"\n--- Testing {url} ---")
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(url)
            print(f"Status: {response.status_code}")
            try:
                data = response.json()
                print(json.dumps(data, indent=2))
            except Exception:
                print(response.text)
    except Exception as e:
        print(f"Error: {e}")

test_endpoint("https://rizzshan-openenv-customer-support.hf.space/health")
test_endpoint("https://rizzshan-openenv-customer-support.hf.space/grader")
test_endpoint("https://rizzshan-openenv-customer-support.hf.space/reset?task_id=easy_refund")
test_endpoint("https://rizzshan-openenv-customer-support.hf.space/grader")
