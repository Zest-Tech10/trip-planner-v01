from crewai import Task
from textwrap import dedent

class TripTasks():
    def __validate_inputs(self, origin, cities, interests, date_range):
        if not origin or not cities or not interests or not date_range:
            raise ValueError("All input parameters must be provided")
        return True

    # 🧭 STEP 1: Deep Market Research (Flights & Hotels)
    def identify_task(self, agent, origin, cities, interests, range):
        self.__validate_inputs(origin, cities, interests, range)
        return Task(
            description=dedent(f"""
                Perform deep market research for a trip from {origin} to {cities} during {range}.
                
                **1. FLIGHTS**: Search for real-world airlines (e.g., Emirates, Air France, Delta) and their 
                   estimated round-trip prices for these specific dates. 
                **2. HOTELS**: Identify 3 specific hotels (e.g., 'Hotel Ritz Paris') with:
                   - Real nightly rates in USD.
                   - Pros/Cons based on the traveler's interests: {interests}.
                **3. CLIMATE**: Provide a specific weather forecast (highs/lows) for {range}.
                **4. SELECTION**: Choose the absolute best city from the list based on this data.

                Output must be a 'Selection Report' with specific names, prices, and reasoning.
            """),
            expected_output="A research report with specific airline names, flight prices, hotel names, and weather data.",
            agent=agent,
            output_key="chosen_city"
        )

    # 🏙️ STEP 2: Local Insight & Verified Pricing
    def gather_task(self, agent, origin, interests, range, context):
        return Task(
            description=dedent(f"""
                Act as a local expert for the 'chosen_city'.
                
                **1. ATTRACTIONS**: List 10 specific venues. For each, find:
                   - The **actual entry fee** in local currency and USD.
                   - Mark clearly as **[FREE]** or **[PAID]**.
                   - Mention if pre-booking is required.
                **2. CULINARY GUIDE**: Identify:
                   - Specific **Must-Eat Dishes** and the **Best Local Restaurants** to find them.
                   - Estimated cost per meal ($, $$, $$$).
                **3. EVENTS**: Search for any specific festivals, concerts, or seasonal markets happening during {range}.
            """),
            expected_output="A comprehensive local guide with verified prices, specific restaurant names, and venue details.",
            agent=agent,
            context=context,
            output_key="city_guide"
        )

    # 🗓️ STEP 3: The Master Travel Plan (Granular Step-by-Step)
    def plan_task(self, agent, origin, interests, range, context):
        return Task(
            description=dedent(f"""
                Create a **complete, day-by-day travel itinerary** for the FULL DURATION of the trip ({range}).
                
                **CRITICAL: CHRONOLOGICAL STEP-BY-STEP FORMAT**:
                For EVERY DAY, provide a granular timeline (e.g., 09:00 AM, 11:30 AM, 01:30 PM):
                1. **Morning (Step 1 & 2)**: What to visit first, including travel time from the hotel.
                2. **Lunch**: Specific restaurant name near the morning activity.
                3. **Afternoon (Step 3 & 4)**: What to visit next, ensuring locations are geographically close.
                4. **Evening**: Dinner at a specific restaurant followed by a night activity or stroll.
                
                **EVERY DAY MUST INCLUDE**:
                - **Location Names**: Real names of parks, museums, and streets.
                - **Transition Instructions**: How to get from 'Place A' to 'Place B' (e.g., "10-min Uber" or "Walk through the park").
                - **Entry Fees**: Mention the cost for each specific stop as it appears in the timeline.

                **YOUR MASTER PLAN MUST ALSO INCLUDE**:
                1. **Logistics Summary**: 
                   - Specific Flight Suggestion (Airline & Total Price).
                   - Selected Hotel with total stay cost.
                   - Arrival Logistics (Airport to Hotel step-by-step).
                2. **Free vs Paid Master Table**: 
                   - A clear Markdown table listing every activity and its cost.
                3. **Complete Budget**: 
                   - Total estimate for Flights + Hotel + Food + Entry Fees.
                4. **Expert Tips**: SIM cards, Tipping, and Local Etiquette.

                Make this a 'Zero-Effort' plan where the traveler just follows the steps.
            """),
            expected_output="A professional, granular, step-by-step travel itinerary with timestamps, transit details, and specific venue names.",
            agent=agent,
            context=context
        )
