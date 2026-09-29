import requests
import json

RS_IDS = [328, 268, 429358]

for rs_id in RS_IDS:
    url = f"https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{rs_id}"

    response = requests.get(url)

    print(f"\n{'=' * 60}")
    print(f"RefSNP rs{rs_id}")
    print(f"HTTP: {response.status_code}")
    print(f"{'=' * 60}")

    if response.ok:
        data = response.json()

        with open(f"data/dbsnp/rs{rs_id}.json", "w") as f:
            json.dump(data, f, indent=2)

        print(f"Guardado en data/dbsnp/rs{rs_id}.json")
    else:
        print(response.text)