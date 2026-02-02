import requests
import json

def check_structure():
    headers = {"User-Agent": "AI-Job-Assistant/1.0"}
    try:
        # Fetch generic feed
        url = "https://remoteok.com/api"
        print(f"Fetching {url}...")
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        
        data = r.json()
        print(f"Total items: {len(data)}")
        
        if len(data) > 1:
            # Item 0 is usually legal/metadata
            print("\nMetadata (Item 0):")
            print(json.dumps(data[0], indent=2))
            
            # Item 1 is a job
            print("\nSample Job (Item 1):")
            job = data[1]
            print(json.dumps(job, indent=2))
            
            print("\nKeys available in job object:")
            print(list(job.keys()))
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_structure()
