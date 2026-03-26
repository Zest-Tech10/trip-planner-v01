import os
import sys
import json
import requests
import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from unstructured.partition.html import partition_html
from langchain_openai import ChatOpenAI

# ---------------------------
# 🔧 LOGGING CONFIGURATION
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class WebsiteInput(BaseModel):
    website: str = Field(..., description="The website URL to scrape")

class BrowserTools(BaseTool):
    name: str = "Scrape website content"
    description: str = "Useful to scrape website content and return the text for analysis"
    args_schema: type[BaseModel] = WebsiteInput

    def _run(self, website: str) -> str:
        try:
            logging.info(f"🌐 Scrapping: {website}")

            api_key = os.getenv("BROWSERLESS_API_KEY")
            if not api_key:
                return "Error: Missing Browserless API key."

            # Fetch content
            url = f"https://chrome.browserless.io/content?token={api_key}"
            payload = json.dumps({"url": website})
            headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}

            response = requests.post(url, headers=headers, data=payload, timeout=15)
            if response.status_code != 200:
                return f"Error: Status {response.status_code}"

            # Extract text
            elements = partition_html(text=response.text)
            content = "\n\n".join([str(el) for el in elements])
            
            # 🚀 OPTIMIZATION: Instead of creating a new CrewAI Agent/Task (slow),
            # we just return the first 15,000 characters. 
            # gpt-4o-mini can handle this easily in its main context.
            if len(content) > 15000:
                return content[:15000] + "\n\n[Content truncated for brevity...]"
            
            logging.info(f"✅ Scraped {len(content)} chars.")
            return content

        except Exception as e:
            logging.error(f"❌ Error: {e}")
            return f"Error: {str(e)}"
