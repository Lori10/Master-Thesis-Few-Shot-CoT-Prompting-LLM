import os
import sys

from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    print("Exported environment variables successfully!")
else:
    print(
        'Please set the environment variables using the keys "OPENAI_API_KEY" in the .env file in the root directory!'
    )
    sys.exit(0)
