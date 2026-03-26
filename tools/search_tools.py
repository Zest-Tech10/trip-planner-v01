import os
import json
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchQuery(BaseModel):
    query: str = Field(..., description="The search query to look up")

class SearchTools(BaseTool):
    name: str = "Search the internet"
    description: str = "Search the internet for travel data (weather, attractions, prices)"
    args_schema: type[BaseModel] = SearchQuery

    def _run(self, query: str) -> str:
        try:
            # 🚀 BALANCED: Return 4 results to ensure we catch flight/hotel prices.
            top_result_to_return = 4
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query})
            headers = {
                'X-API-KEY': os.getenv('SERPER_API_KEY'),
                'content-type': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
            
            if response.status_code != 200:
                return f"Error: Status {response.status_code}"
            
            data = response.json()
            results = data.get('organic', [])
            string = []
            for result in results[:top_result_to_return]:
                string.append(f"Title: {result.get('title')}\nLink: {result.get('link')}\nSnippet: {result.get('snippet')}\n---")
            
            return '\n'.join(string) if string else "No results found"
        except Exception as e:
            return f"Error: {str(e)}"
