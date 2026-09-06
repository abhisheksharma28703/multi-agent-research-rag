import sys
import os
import warnings

# Suppress benign third-party library notices
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

print(f"[SUCCESS] GOOGLE_API_KEY found! (Length: {len(api_key)})")
print("Connecting to Gemini...")

try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    response = llm.invoke("Say hello and confirm you are ready!")
    
    # Extract plain text content cleanly
    reply_text = response.content
    if isinstance(reply_text, list) and len(reply_text) > 0 and isinstance(reply_text[0], dict):
        reply_text = reply_text[0].get("text", reply_text)

    print("\n--- Gemini Response ---")
    print(reply_text)
    print("\n[SUCCESS] Connection to Gemini is 100% verified and working!")
except Exception as e:
    print(f"\n[ERROR] Could not connect to Gemini: {e}")
