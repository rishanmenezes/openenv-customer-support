import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def test_endpoint(url):
    print(f"\n--- Testing {url} ---")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=20) as response:
            body = response.read().decode('utf-8')
            try:
                data = json.loads(body)
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print(body)
    except Exception as e:
        print(f"Error: {e}")

test_endpoint("https://rizzshan-openenv-customer-support.hf.space/health")
test_endpoint("https://rizzshan-openenv-customer-support.hf.space/grader")
test_endpoint("https://rizzshan-openenv-customer-support.hf.space/reset?task_id=easy_refund")
test_endpoint("https://rizzshan-openenv-customer-support.hf.space/grader")
