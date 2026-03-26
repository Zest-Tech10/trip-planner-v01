import sys
import os
import logging
from crewai import Agent, LLM
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

class TripAgents():
    def __init__(self, llm: LLM = None):
        # 🚀 OPTIMIZATION: Default to gpt-4o-mini if no LLM provided.
        # It's 10x faster than gpt-4 and perfect for these agents.
        if llm is None:
            self.llm = LLM(model="gpt-4o-mini", temperature=0.7)
        else:
            self.llm = llm

        # Initialize tools
        self.search_tool = SearchTools()
        self.browser_tool = BrowserTools()
        self.calculator_tool = CalculatorTools()
        logging.info("✅ TripAgents Initialized with Fast LLM")

    def city_selection_agent(self):
        return Agent(
            role='City Selection Expert',
            goal='Select the best city based on weather, season, and prices',
            backstory='Expert in analyzing travel data to pick ideal destinations',
            tools=[self.search_tool, self.browser_tool],
            allow_delegation=False,
            llm=self.llm,
            verbose=False,
            cache=True
        )

    def local_expert(self):
        return Agent(
            role='Local Expert at this city',
            goal='Provide the BEST insights about the selected city',
            backstory="A knowledgeable local guide with extensive information",
            tools=[self.search_tool, self.browser_tool],
            allow_delegation=False,
            llm=self.llm,
            verbose=False,
            cache=True
        )

    def travel_concierge(self):
        return Agent(
            role='Amazing Travel Concierge',
            goal="Create amazing itineraries with budget and packing suggestions",
            backstory="Specialist in travel planning and logistics",
            tools=[self.search_tool, self.browser_tool, self.calculator_tool],
            allow_delegation=False,
            llm=self.llm,
            verbose=False,
            cache=True
        )


# python Trip_Final.py