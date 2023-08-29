import os
import sys

from dotenv import dotenv_values

env_vars_azureopenai = dotenv_values(".env_azureopenai")
azureopenai_api_key = env_vars_azureopenai.get("OPENAI_API_KEY")

if azureopenai_api_key:
    print("Exported environment variables successfully!")
else:
    print(
        'Please set the environment variables using the keys "OPENAI_API_KEY" in the .env.azureopenai file in the root directory!'
    )
    sys.exit(0)
