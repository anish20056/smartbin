"""
deploy.py
=========
Starts the Smart Bin API and creates a public URL using ngrok.
Anyone on any device can access the Smart Bin UI from the public URL.

Run:
    python deploy.py

Requirements:
    pip install pyngrok uvicorn fastapi
"""

import threading
import time
import webbrowser
from pathlib import Path

# ── Check pyngrok ──────────────────────────────────────────────────────────────
try:
    from pyngrok import ngrok, conf
except ImportError:
    print("Installing pyngrok...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
    from pyngrok import ngrok, conf

import uvicorn

# ── Config ─────────────────────────────────────────────────────────────────────
PORT         = 8000
HTML_FILE    = "smartbin_ui.html"
OUTPUT_HTML  = "smartbin_ui_deployed.html"

# ── Start API in background thread ─────────────────────────────────────────────
def start_api():
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",   # quieter logs
    )

print("\n" + "="*55)
print("  SMART BIN AI — Deployment")
print("="*55)

# Start API server in background
print("\n⚙️  Starting API server...")
api_thread = threading.Thread(target=start_api, daemon=True)
api_thread.start()
time.sleep(3)  # Wait for server to be ready
print(f"✅ API running on http://localhost:{PORT}")

# ── Start ngrok tunnel ─────────────────────────────────────────────────────────
print("\n🌐 Creating public tunnel via ngrok...")
try:
    tunnel     = ngrok.connect(PORT, "http")
    public_url = tunnel.public_url
    print(f"✅ Public URL: {public_url}")
except Exception as e:
    print(f"\n❌ ngrok failed: {e}")
    print("\nFallback — using localhost only.")
    print(f"Open smartbin_ui.html in your browser.")
    print(f"API is running at http://localhost:{PORT}")
    print("\nPress Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
    exit()

# ── Update HTML with public URL ────────────────────────────────────────────────
print(f"\n📄 Creating deployed HTML file...")
try:
    html = Path(HTML_FILE).read_text(encoding="utf-8")
    html = html.replace("http://localhost:8000", public_url)
    html = html.replace("http://127.0.0.1:8000", public_url)
    Path(OUTPUT_HTML).write_text(html, encoding="utf-8")
    print(f"✅ Created: {OUTPUT_HTML}")
except FileNotFoundError:
    print(f"⚠️  {HTML_FILE} not found — HTML not updated.")
    print("   Make sure smartbin_ui.html is in the same folder.")

# ── Print instructions ─────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  🚀 SMART BIN IS LIVE")
print("="*55)
print(f"\n  API URL  : {public_url}")
print(f"  Local URL: http://localhost:{PORT}")
print(f"\n  👉 Open '{OUTPUT_HTML}' in your browser")
print(f"  👉 Or share the public URL with anyone")
print(f"\n  Press Ctrl+C to stop\n")
print("="*55 + "\n")

# Open the deployed HTML automatically
try:
    webbrowser.open(OUTPUT_HTML)
    print("✅ Opened browser automatically")
except:
    pass

# ── Keep alive ─────────────────────────────────────────────────────────────────
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n🛑 Shutting down...")
    ngrok.kill()
    print("✅ Stopped. Goodbye!\n")
