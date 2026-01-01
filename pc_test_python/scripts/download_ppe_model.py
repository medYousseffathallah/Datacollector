import os
import requests
import sys

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def main():
    # Go up one level from scripts/ to project root, then into models/
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    dest_path = os.path.join(model_dir, 'ppe_best.pt')
    
    if os.path.exists(dest_path):
        print(f"Model already exists at {dest_path}")
        return

    # Try main branch first
    url_main = "https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection/raw/main/models/best.pt"
    if download_file(url_main, dest_path):
        return

    # Try master branch if main fails
    url_master = "https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection/raw/master/models/best.pt"
    if download_file(url_master, dest_path):
        return
    
    print("Failed to download model from both main and master branches.")
    sys.exit(1)

if __name__ == "__main__":
    main()
