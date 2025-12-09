from pathlib import Path

from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env")
