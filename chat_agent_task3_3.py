#!/usr/bin/env python3
"""
Interactive Chat Interface for Task 3.3 Planning Agent with SQLite Persistence

Conversation flow:
1. Collect the four key trip constraints (destination, origin, duration, budget).
2. Validate that the information is complete and sensible.
3. Ask for explicit confirmation before calling `agent.create_plan`.
4. Persist constraints and conversation history in SQLite.

Usage:
    python chat_agent.py [--session-id SESSION_ID] [--db-path PATH]
"""

import asyncio
import json
import os
import sqlite3
import uuid
import argparse
from datetime import date, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
from backend.models.agent_schemas import PlanRequest, TripConstraints
from backend.services.planning_agent import get_planning_agent
from openai import OpenAI

# Load .env if it exists next to this file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # Dotenv not installed, ignore optional configuration


class SessionManager:
    """SQLite-backed storage for session constraints and message history."""

    def __init__(self, db_path: str = "chat_sessions.sqlite3"):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create database tables if they do not already exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_state (
                    session_id TEXT PRIMARY KEY,
                    constraints_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_message (
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def load_constraints(self, session_id: str) -> Optional[TripConstraints]:
        """Fetch persisted constraints for the given session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT constraints_json FROM session_state WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                try:
                    data = json.loads(row[0])
                    return TripConstraints(**data)
                except Exception:
                    return None
        return None

    def save_constraints(self, session_id: str, constraints: TripConstraints):
        """Persist constraints for the given session."""
        constraints_json = constraints.model_dump_json()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO session_state (session_id, constraints_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (session_id, constraints_json))
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str):
        """Append a chat message to the conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO session_message (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )
            conn.commit()

    def load_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Retrieve recent chat messages in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """SELECT role, content FROM session_message
                   WHERE session_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, limit)
            )
            messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
            return list(reversed(messages))


def parse_constraints_from_text(text: str, existing: Optional[TripConstraints] = None) -> TripConstraints:
    """Extract constraint information from user text."""
    import re

    constraints = existing or TripConstraints()

    # Extract budget, prioritising currency-aware patterns first
    money_patterns = [
        # Highest priority: patterns with explicit currency
        (r'(?P<currency>NZ)\s*\$\s*(?P<amount>\d+)', 'NZD'),  # "NZ$1000"
        (r'(?P<currency>NZD)\s*(?P<amount>\d+)', 'NZD'),  # "NZD 1000"
        (r'budget\s*(?:is|of)?\s*(?P<currency_word>NZD|NZ)\s*\$?\s*(?P<amount>\d+)', 'NZD'),
        (r'(?P<currency>USD|US)\s*\$?\s*(?P<amount>\d+)', 'USD'),
        (r'(?P<currency>AUD|AU)\s*\$?\s*(?P<amount>\d+)', 'AUD'),
        (r'£\s*(?P<amount>\d+)', 'GBP'),
        (r'€\s*(?P<amount>\d+)', 'EUR'),
        (r'budget\s*(?:is|of)?\s*\$?\s*(?P<amount>\d+)', None),
        (r'under\s*\$?\s*(?P<amount>\d+)', None),
        (r'\$\s*(?P<amount>\d+)', None),
        (r'with\s+(?P<amount>\d+)', None),
        (r'\b(?P<amount>\d{2,5})\b', None),  # Fallback: plain numbers
    ]

    for pattern, forced_currency in money_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            groups = match.groupdict()
            amount = groups.get("amount")
            if amount:
                constraints.budget = float(amount)
                if forced_currency:
                    constraints.currency = forced_currency
                else:
                    cur = groups.get("currency") or groups.get("currency_word")
                    if cur:
                        cur_upper = cur.upper()
                        if "NZ" in cur_upper:
                            constraints.currency = "NZD"
                        elif "US" in cur_upper:
                            constraints.currency = "USD"
                        elif "AU" in cur_upper:
                            constraints.currency = "AUD"
                    elif not constraints.currency:
                        constraints.currency = "USD"
                break

    # Extract trip length (days or weeks)
    day_patterns = [
        (r'(\d+)[-\s]days?', 1),
        (r'for\s+(\d+)\s+days?', 1),
        (r'(\d+)\s+day\s+trip', 1),
        (r'(\d+)[-\s]weeks?', 7),  # Convert weeks to days
        (r'for\s+(\d+)\s+weeks?', 7),
    ]
    for pattern, multiplier in day_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            constraints.days = int(match.group(1)) * multiplier
            break

    # Derive destination and origin cities
    # Quick list of common cities to speed up matching (non-exhaustive)
    common_cities = ['Auckland', 'Wellington', 'Christchurch', 'London', 'Paris', 'Tokyo',
                     'New York', 'Sydney', 'Singapore', 'Rome', 'Barcelona', 'Beijing',
                     'Shanghai', 'Guangzhou', 'Shenzhen', 'Hong Kong', 'Taipei']

    # First scan for the common city names (case insensitive)
    for city in common_cities:
        if city.lower() in text.lower():
            # Decide whether this city is destination or origin
            if re.search(rf'\b(?:to|visit|in|go)\s+{city}', text, re.IGNORECASE):
                constraints.destination_city = city
            elif re.search(rf'\b(?:from|leaving|depart)\s+{city}', text, re.IGNORECASE):
                constraints.origin_city = city
            elif not constraints.destination_city and not constraints.origin_city:
                # Default to destination when no explicit to/from indicator
                constraints.destination_city = city

    # Generic to/from/go patterns to catch other cities
    to_patterns = [
        r'\b(?:to|visit|visiting|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "to Zhuhai"
        r'\bgo\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "go Auckland"
    ]

    for pattern in to_patterns:
        to_match = re.search(pattern, text)
        if to_match and not constraints.destination_city:
            city_name = to_match.group(1).strip()
            # Ignore common non-city words
            if city_name.lower() not in ['with', 'for', 'from', 'days', 'day', 'week', 'weeks']:
                constraints.destination_city = city_name
                break

    from_patterns = [
        r'\b(?:from|leaving|depart(?:ing)?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # "from Zhuhai"
        r'\b([A-Z][a-z]+)\s+to\s+[A-Z]',  # "Zhuhai to Tokyo" pattern
    ]

    for pattern in from_patterns:
        from_match = re.search(pattern, text)
        if from_match and not constraints.origin_city:
            city_name = from_match.group(1).strip()
            # Ignore common non-city words
            if city_name.lower() not in ['with', 'for', 'to', 'days', 'day', 'week', 'weeks', 'go', 'plan']:
                constraints.origin_city = city_name
                break

    # Extract preference keywords
    preferences = list(constraints.preferences or [])
    pref_keywords = {
        'museum': 'museums',
        'beach': 'beaches',
        'history': 'history',
        'food': 'food',
        'outdoor': 'outdoor activities',
        'budget': 'budget-friendly',
        'luxury': 'luxury',
        'adventure': 'adventure',
        'culture': 'cultural experiences'
    }
    for keyword, pref in pref_keywords.items():
        if keyword in text.lower() and pref not in preferences:
            preferences.append(pref)

    if preferences:
        constraints.preferences = preferences

    return constraints


def llm_extract_constraints(text: str, existing: Optional[TripConstraints] = None) -> TripConstraints:
    """Fallback extraction that uses an LLM when heuristics fail."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, skipping LLM extraction")
            return existing or TripConstraints()

        client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        existing_json = existing.model_dump() if existing else {}

        # Highlight missing fields for the LLM
        missing_fields = []
        if not existing_json.get('destination_city'):
            missing_fields.append('destination')
        if not existing_json.get('origin_city'):
            missing_fields.append('origin')

        context_hint = ""
        if missing_fields:
            context_hint = f"\nContext: We are asking the user about: {', '.join(missing_fields)}"

        prompt = f"""Extract trip planning information from the user's message. Return ONLY a JSON object.

User message: "{text}"{context_hint}

Current information: {json.dumps(existing_json, ensure_ascii=False)}

Extract and return JSON with these fields (keep existing values if not mentioned):
{{
    "destination_city": "destination city name or null",
    "origin_city": "origin city name or null",
    "days": number of days or null,
    "budget": number or null,
    "currency": "USD/NZD/AUD/GBP/EUR or null"
}}

Rules:
- Any city name worldwide is valid (e.g., Zhuhai, Auckland, Beijing, Macau, etc.)
- If user provides a single city name when origin is missing, assume it's the origin city
- If user says "1 week", convert to days: 7
- Default currency is USD if $ is mentioned without specification
- Return null for missing information
- ONLY return the JSON object, no explanation"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()

        # Strip potential Markdown wrappers
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        extracted = json.loads(result_text)

        # Merge details into the existing constraint object
        constraints = existing or TripConstraints()
        if extracted.get("destination_city"):
            constraints.destination_city = extracted["destination_city"]
        if extracted.get("origin_city"):
            constraints.origin_city = extracted["origin_city"]
        if extracted.get("days"):
            constraints.days = int(extracted["days"])
        if extracted.get("budget"):
            constraints.budget = float(extracted["budget"])
        if extracted.get("currency"):
            constraints.currency = extracted["currency"]

        return constraints

    except Exception as e:
        # Keep original constraints if the LLM call fails
        print(f"⚠️  LLM extraction failed: {e}")
        return existing or TripConstraints()


def has_potential_missing_info(text: str) -> bool:
    """Heuristically check whether text may contain unparsed location hints."""
    import re

    # 1. Look for capitalised words that could be city names
    capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
    common_words = {'I', 'The', 'A', 'An', 'My', 'We', 'He', 'She', 'It', 'They', 'This', 'That', 'From', 'To'}
    potential_cities = [w for w in capitalized_words if w not in common_words]

    if len(potential_cities) > 0:
        return True

    # 2. If there are no capitalised words, check for a single lowercase city (e.g., "zhuhai")
    words = text.strip().split()
    if len(words) == 1 and words[0].isalpha() and len(words[0]) >= 3:
        # Skip common English words
        common_lowercase = {'from', 'to', 'yes', 'no', 'ok', 'sure', 'go', 'the', 'and', 'or', 'but', 'for', 'with'}
        if words[0].lower() not in common_lowercase:
            return True

    return False


def extract_constraints_hybrid(text: str, existing: Optional[TripConstraints] = None) -> tuple[TripConstraints, bool]:
    """
    Hybrid extraction: use regex first, then fall back to an LLM if needed.

    Returns:
        (constraints, used_llm): Updated constraints and whether the LLM was used.
    """
    # 1. Start with the lightweight regex approach
    constraints = parse_constraints_from_text(text, existing)

    # 2. Decide if an LLM fallback is warranted
    old_constraints = existing or TripConstraints()

    # No new city info yet, but text might still contain one
    new_dest = constraints.destination_city != old_constraints.destination_city
    new_origin = constraints.origin_city != old_constraints.origin_city
    has_new_city = new_dest or new_origin

    # Regex missed a probable city mention
    if not has_new_city and has_potential_missing_info(text):
        # 2a. Quick fix: capitalise a single lowercase word
        words = text.strip().split()
        if len(words) == 1 and words[0].isalpha():
            capitalized = words[0].capitalize()
            constraints = parse_constraints_from_text(capitalized, existing)
            new_dest = constraints.destination_city != old_constraints.destination_city
            new_origin = constraints.origin_city != old_constraints.origin_city
            if new_dest or new_origin:
                return constraints, False  # Regex extraction now succeeded

        # 2b. Still nothing—ask the LLM
        print("🤖 Using LLM to extract city names...")
        constraints = llm_extract_constraints(text, constraints)
        return constraints, True

    return constraints, False


def check_constraints_complete(constraints: TripConstraints) -> tuple[bool, List[str]]:
    """Check whether all four core constraints are present."""
    missing = []
    if not constraints.destination_city:
        missing.append("destination")
    if not constraints.origin_city:
        missing.append("origin")
    if not constraints.days or constraints.days < 1:
        missing.append("days")
    if not constraints.budget or constraints.budget < 0:
        missing.append("budget")

    return len(missing) == 0, missing


def validate_constraints(constraints: TripConstraints) -> tuple[bool, List[str]]:
    """Validate constraint sanity and return any issues."""
    issues = []

    # Validate trip length
    if constraints.days and (constraints.days < 1 or constraints.days > 30):
        issues.append(f"Trip duration {constraints.days} days seems unrealistic (should be 1-30)")

    # Validate budget range
    if constraints.budget:
        if constraints.budget < 100:
            issues.append(f"Budget {constraints.currency} {constraints.budget} seems too low")
        elif constraints.budget > 50000:
            issues.append(f"Budget {constraints.currency} {constraints.budget} seems very high")

    # Detect local trips (origin equals destination)
    if constraints.origin_city and constraints.destination_city:
        if constraints.origin_city.lower().strip() == constraints.destination_city.lower().strip():
            issues.append("Origin and destination are the same (local trip - no flights needed)")

    return len(issues) == 0, issues


def format_constraints_summary(constraints: TripConstraints) -> str:
    """Render a human-readable summary of the collected constraints."""
    lines = []
    if constraints.destination_city:
        lines.append(f"📍 Destination: {constraints.destination_city}")
    if constraints.origin_city:
        lines.append(f"🛫 Origin: {constraints.origin_city}")
    if constraints.days:
        lines.append(f"📅 Duration: {constraints.days} days")
    if constraints.budget:
        lines.append(f"💰 Budget: {constraints.currency} {constraints.budget}")
    if constraints.preferences:
        lines.append(f"❤️  Preferences: {', '.join(constraints.preferences)}")

    return "\n".join(lines) if lines else "No information collected yet"


async def chat_with_agent(session_id: str, db_path: str):
    """Primary interactive loop for the smart planning agent."""

    print("=" * 80)
    print("💬 Smart Planning Agent Chat Interface")
    print("=" * 80)
    print()
    print(f"Session ID: {session_id}")
    print()
    print("Hello! I'm your AI trip planning assistant.")
    print("I'll help you plan your trip by collecting key information first:")
    print("  • Where do you want to go?")
    print("  • Where are you leaving from?")
    print("  • How many days?")
    print("  • What's your budget?")
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 80)
    print()

    # Initialise persistence layer
    session_mgr = SessionManager(db_path)

    # Restore saved constraints if they exist
    constraints = session_mgr.load_constraints(session_id) or TripConstraints()

    # Replay the previous conversation
    history = session_mgr.load_messages(session_id)
    if history:
        print(f"📜 Loaded {len(history)} previous messages")
        print()

    # Show any remembered constraints for quick reference
    if constraints.destination_city or constraints.origin_city or constraints.days or constraints.budget:
        print("📝 Previously saved information:")
        print(format_constraints_summary(constraints))
        print()

    # Lazy-load the planning agent
    agent = None

    while True:
        # Collect user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\n👋 Goodbye! Safe travels!")
            break

        # Persist user message
        session_mgr.add_message(session_id, "user", user_input)

        print()

        # Handle diagnostic commands
        if user_input.lower() in ['status', 'info', 'show']:
            print("📝 Current trip information:")
            print(format_constraints_summary(constraints))
            is_complete, missing = check_constraints_complete(constraints)
            if not is_complete:
                print(f"\n⚠️  Missing: {', '.join(missing)}")
            else:
                is_valid, issues = validate_constraints(constraints)
                if not is_valid:
                    print(f"\n⚠️  Issues: {', '.join(issues)}")
                else:
                    print("\n✅ All information complete and valid!")
            print()
            continue

        if user_input.lower() in ['reset', 'clear', 'restart']:
            constraints = TripConstraints()
            session_mgr.save_constraints(session_id, constraints)
            print("🔄 Trip information cleared. Let's start fresh!")
            print()
            continue

        # Extract constraints using hybrid strategy
        old_constraints = constraints.model_copy(deep=True)
        constraints, used_llm = extract_constraints_hybrid(user_input, constraints)

        # Detect new information since the last turn
        has_new_info = (
            constraints.destination_city != old_constraints.destination_city or
            constraints.origin_city != old_constraints.origin_city or
            constraints.days != old_constraints.days or
            constraints.budget != old_constraints.budget
        )

        # Persist and display updated constraints
        if has_new_info:
            session_mgr.save_constraints(session_id, constraints)
            print("✅ Got it! Here's what I understand:")
            print(format_constraints_summary(constraints))
            print()

        # Verify whether all four constraints are captured
        is_complete, missing = check_constraints_complete(constraints)

        if not is_complete:
            # Still missing fields—ask targeted follow-up questions
            if not has_new_info:
                # Gently remind the user when no new data was provided
                print("🤔 I'd love to help! To plan your trip, I need some information.")
            print(f"⚠️  Still need: {', '.join(missing)}")
            print()
            if "destination" in missing:
                print("Where would you like to go?")
            elif "origin" in missing:
                print("Which city will you be departing from?")
            elif "days" in missing:
                print("How many days will your trip last?")
            elif "budget" in missing:
                print("What's your total budget? (e.g., 500 NZD, $1000)")
            print()

            # Log assistant guidance
            response = f"Still need: {', '.join(missing)}"
            session_mgr.add_message(session_id, "assistant", response)
            continue

        # Constraints present—validate them
        is_valid, issues = validate_constraints(constraints)

        if not is_valid:
            print("⚠️  I noticed some potential issues:")
            for issue in issues:
                print(f"   • {issue}")
            print()
            print("Would you like to proceed anyway, or update the information?")
            print("(Type 'yes' to proceed, or provide updated information)")
            print()

            # Log assistant message summarising issues
            response = f"Issues found: {', '.join(issues)}"
            session_mgr.add_message(session_id, "assistant", response)
            continue

        # Ask for confirmation before calling the planner
        if has_new_info or "plan" in user_input.lower() or "ready" in user_input.lower():
            print("✅ Perfect! I have all the information:")
            print(format_constraints_summary(constraints))
            print()
            print("Ready to create your trip plan? (Type 'yes' to start, or update any information)")
            print()

            # Capture confirmation response
            try:
                confirmation = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            session_mgr.add_message(session_id, "user", confirmation)

            if confirmation.lower() not in ['yes', 'y', 'sure', 'ok', 'okay', 'go', 'proceed']:
                print()
                print("No problem! Let me know what you'd like to change.")
                print()
                continue

        # Generate the travel plan
        print()
        print("🔄 Great! Let me plan your trip...")
        print()

        # Fill default start/end dates if needed
        if not constraints.start_date:
            start = date.today() + timedelta(days=7)
            constraints.start_date = start.isoformat()
            if constraints.days:
                constraints.end_date = (start + timedelta(days=constraints.days - 1)).isoformat()

        # Defensively create the agent just-in-time
        if agent is None:
            agent = get_planning_agent()
            print(f"🤖 Agent ready (using {agent.model_name})")
            print()

        try:
            # Build the planning request
            request = PlanRequest(
                prompt=f"Plan a {constraints.days}-day trip to {constraints.destination_city}",
                constraints=constraints,
                max_iterations=5
            )

            # Call the planning agent
            print("🔄 Planning in progress (calling tools: flights, weather, attractions)...")
            print()

            response = await agent.create_plan(request)

            # Present the structured plan
            print("-" * 80)
            print("✅ Here's your trip plan!")
            print("-" * 80)
            print()

            # Destination summary
            print(f"📍 Destination: {response.itinerary.destination}")
            print()

            # Flights snippet
            if response.itinerary.flights:
                print("✈️  Flights:")
                for i, flight in enumerate(response.itinerary.flights[:2], 1):
                    print(f"   {i}. {flight.airline} {flight.flight_number}")
                    print(f"      {flight.departure_time} → {flight.arrival_time}")
                    print(f"      Price: {response.itinerary.currency} {flight.price}")
                    print(f"      Duration: {flight.duration_hours} hours")
                    print()

            # Weather snippet
            if response.itinerary.weather_forecast:
                print("🌤️  Weather Forecast:")
                for day in response.itinerary.weather_forecast[:3]:
                    print(f"   {day.date}: {day.temperature_celsius}°C, {day.condition}")
                print()

            # Attractions snippet
            if response.itinerary.attractions:
                print("🎯 Top Attractions:")
                for i, attr in enumerate(response.itinerary.attractions[:5], 1):
                    print(f"   {i}. {attr.name} ({attr.category}) - ⭐ {attr.rating}/5")
                    print(f"      {attr.price_range}")
                print()

            # Cost breakdown
            print("💰 Cost Breakdown:")
            flight_cost = sum(f.price for f in response.itinerary.flights)
            if flight_cost > 0:
                print(f"   • Flights: {response.itinerary.currency} {flight_cost:.2f}")

            num_days = len(response.itinerary.daily_plan) if response.itinerary.daily_plan else 0
            if num_days > 0:
                accommodation = 100 * (num_days - 1) if num_days > 1 else 0
                meals = 50 * num_days
                other = 35 * num_days

                if accommodation > 0:
                    print(f"   • Accommodation: ~{response.itinerary.currency} {accommodation:.2f}")
                print(f"   • Meals: ~{response.itinerary.currency} {meals:.2f}")
                print(f"   • Transport & Activities: ~{response.itinerary.currency} {other:.2f}")

            print(f"   ─────────────────────────")
            print(f"   💵 Total: {response.itinerary.currency} {response.itinerary.total_cost:.2f}")
            print()

            # Constraint satisfaction report
            if response.constraints_satisfied:
                print("✅ All constraints satisfied!")
            else:
                print("⚠️  Constraint issues:")
                for violation in response.constraint_violations:
                    print(f"   • {violation}")
            print()

            print(f"🔧 Tools used: {len(response.tool_calls)} calls in {response.total_iterations} iterations")
            print(f"⏱️  Planning time: {response.planning_time_ms:.0f}ms")
            print()
            print("-" * 80)
            print()

            # Archive assistant summary
            plan_summary = f"Created trip plan: {response.itinerary.destination}, {num_days} days, {response.itinerary.currency} {response.itinerary.total_cost:.2f}"
            session_mgr.add_message(session_id, "assistant", plan_summary)

            # Offer adjustments
            print("Would you like to make any changes to this plan? (or type 'quit' to exit)")
            print()

        except Exception as e:
            print(f"❌ Sorry, I encountered an error: {e}")
            print()
            session_mgr.add_message(session_id, "assistant", f"Error: {e}")


async def main():
    """Command-line entrypoint."""
    parser = argparse.ArgumentParser(description="Smart Planning Agent Chat")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Session ID (default: generate new UUID)")
    parser.add_argument("--db-path", type=str, default="chat_sessions.sqlite3",
                        help="Path to SQLite database (default: chat_sessions.sqlite3)")

    args = parser.parse_args()

    session_id = args.session_id or str(uuid.uuid4())[:8]
    db_path = args.db_path

    await chat_with_agent(session_id, db_path)


if __name__ == "__main__":
    asyncio.run(main())
