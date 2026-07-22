"""
Single entry point that starts the whole Voss Mobility Dispatch
service WITHOUT Docker
"""

import os
import sys
import socket
import urllib.request
import urllib.error


def check_port_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_ollama() -> bool:
    """Ollama's local API listens on 11434 by default."""
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except urllib.error.URLError:
        return False
    except Exception:
        return check_port_open("localhost", 11434)


def check_model_file() -> bool:
    return os.path.isfile(os.path.join("models", "demand_model.pkl"))


def check_env_file() -> bool:
    return os.path.isfile(".env")


def main():
    print("=" * 55)
    print(" Voss Mobility - Autonomous Ride Dispatch Agent")
    print(" Starting service (no Docker required)")
    print("=" * 55)

    problems = []

    print("\n[1/4] Checking trained model...", end=" ")
    if check_model_file():
        print("OK")
    else:
        print("MISSING")
        problems.append(
            "models/demand_model.pkl not found. Run `python train_model.py` first."
        )

    print("[2/4] Checking database credentials (.env)...", end=" ")
    if check_env_file():
        print("OK")
    else:
        print("MISSING")
        problems.append(
            ".env file not found. Create one with DB_HOST, DB_PORT, DB_NAME, "
            "DB_USER, DB_PASSWORD."
        )

    print("[3/4] Checking Ollama (local LLM engine)...", end=" ")
    if check_ollama():
        print("OK")
    else:
        print("NOT DETECTED")
        problems.append(
            "Ollama does not appear to be running on localhost:11434. "
            "Start Ollama, and make sure the 'mistral' model has been "
            "pulled (`ollama run mistral`)."
        )

    frontend_dist = os.path.join("frontend", "dist")
    print("[4/4] Checking for built frontend...", end=" ")
    if os.path.isdir(frontend_dist):
        print(f"OK ({frontend_dist})")
    else:
        print("NOT BUILT")
        print(
            "        (Optional) Run `npm run build` inside frontend/ to serve "
            "the dashboard from this same service. Continuing with API only."
        )

    if problems:
        print("\nSome checks failed:")
        for p in problems:
            print(f"  - {p}")
        print("\nYou can still continue, but some features may not work.")
        answer = input("Continue anyway? [y/N]: ").strip().lower()
        if answer != "y":
            print("Exiting.")
            sys.exit(1)

    print("\nStarting server on http://127.0.0.1:8000 ...")
    print("API docs available at http://127.0.0.1:8000/docs")
    if os.path.isdir(frontend_dist):
        print("Dashboard available at http://127.0.0.1:8000/")
    print("Press CTRL+C to stop.\n")

    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
