"""Each tool one by one against faked services, plus the Groq -> Gemini provider chain."""
from __future__ import annotations

import pytest
import json
import time
from langchain_core.messages import AIMessage

import tools.foursquare_api as fsq
import tools.mapbox_api as mapbox
import tools.travel_market as market
import tools.weather as weather

REAL_OSRM = mapbox._osrm  # conftest swaps mapbox._osrm for a fake; these tests need the real one
import llm


# ====================================================================================================
# Tools
# ====================================================================================================

# --- Mapbox ----------------------------------------------------------------------------------------------------------

def test_geocode_location_mapbox_then_openstreetmap_then_not_found(monkeypatch):
    mapbox._geocode_cache.clear()
    found = mapbox.geocode_location.invoke({"place_name": "Munnar, India"})  # the fake world answers as OpenStreetMap
    assert found["source"] == "osm" and 9.5 < found["lat"] < 10.5
    monkeypatch.setattr(mapbox, "MAPBOX_API_KEY", "pk.test")
    monkeypatch.setattr(mapbox.httpx, "get", lambda *a, **k: type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: {"features": [{"center": [77.06, 10.09], "text": "Munnar"}]}})())
    mapbox._geocode_cache.clear()
    assert mapbox.geocode_location.invoke({"place_name": "Munnar, India"})["source"] == "mapbox"
    monkeypatch.setattr(mapbox, "_nominatim", lambda p: None)
    monkeypatch.setattr(mapbox.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    mapbox._geocode_cache.clear()
    miss = mapbox.geocode_location.invoke({"place_name": "Nowhere At All"})
    assert miss["lat"] is None and miss["source"] == "not_found"
    mapbox._geocode_cache.clear()


def test_get_directions_returns_road_time_and_cost():
    mapbox._directions_cache.clear()
    out = mapbox.get_directions.invoke({"origin": "Kochi", "destination": "Munnar"})
    assert out["mode"] == "road" and out["duration_hours"] > 0 and out["cost_inr"] > 0 and out["distance_km"] > 0
    mapbox._directions_cache.clear()


def test_osrm_answers_when_mapbox_does_not_and_its_times_are_stretched_to_real_roads(monkeypatch):
    """Mapbox blocked: OpenStreetMap's road distance is used, with OSRM's free-flow time x1.4."""
    mapbox._directions_cache.clear()
    monkeypatch.setattr(mapbox, "_osrm", REAL_OSRM)  # the fake world replaces it in other tests
    monkeypatch.setattr(mapbox.httpx, "get", lambda *a, **k: type("R", (), {"raise_for_status": lambda s: None, "json": lambda s: {"routes": [{"distance": 84000, "duration": 5400}]}})())
    out = mapbox.get_directions.invoke({"origin": "Munnar", "destination": "Thekkady"})
    assert out["source"] == "osrm" and out["distance_km"] == 84.0 and out["duration_hours"] == pytest.approx(1.5 * 1.4)
    mapbox._directions_cache.clear()


def test_drive_times_are_reported_unavailable_not_invented_when_every_service_is_down(monkeypatch):
    mapbox._directions_cache.clear()
    monkeypatch.setattr(mapbox, "_osrm", REAL_OSRM)
    monkeypatch.setattr(mapbox.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = mapbox.get_directions.invoke({"origin": "Munnar", "destination": "Thekkady"})
    assert out["source"] == "unavailable" and out["duration_hours"] == 0.0
    assert "Thekkady" not in {k[1] for k in mapbox._directions_cache}, "a failed lookup is not remembered"


def test_travel_time_matrix_is_square_with_a_zero_diagonal():
    pts = [{"lat": 10.0, "lng": 77.0}, {"lat": 10.05, "lng": 77.02}, {"lat": 10.1, "lng": 77.1}]
    out = mapbox.travel_time_matrix.invoke({"points": pts})
    m = out["matrix"]
    assert len(m) == 3 and all(len(r) == 3 for r in m) and all(m[i][i] == 0 for i in range(3)) and m[0][2] > m[0][1]
    assert mapbox.travel_time_matrix.invoke({"points": []})["matrix"] == []


# --- places (LLM), Foursquare helpers ------------------------------------------------------------------------------------

def test_a_geocoder_that_finds_a_same_named_village_elsewhere_is_not_trusted(monkeypatch):
    """Mapbox put Goa's Palolem in central India. The model's coordinates win when the two disagree by >60 km."""
    fsq._destination_cache.clear()
    monkeypatch.setattr(fsq, "llm_json", lambda s, u: {"places": [
        {"name": "Palolem", "lat": 15.01, "lng": 74.02}, {"name": "Panaji", "lat": 15.50, "lng": 73.83}]})
    monkeypatch.setattr(fsq, "geocode_location", type("G", (), {"invoke": staticmethod(lambda a: (
        {"lat": 22.2, "lng": 78.5, "source": "mapbox"} if a["place_name"].startswith("Palolem") else {"lat": 15.498, "lng": 73.828, "source": "mapbox"}))}))
    places = {p["name"]: p for p in fsq.search_destinations.invoke({"region": "Goa, India"})}
    assert places["Palolem"]["lat"] == 15.01, "the far-away geocode must be ignored"
    assert places["Panaji"]["lat"] == 15.498, "a nearby geocode refines the model's coordinates"
    fsq._destination_cache.clear()


def test_search_destinations_lists_the_models_places(monkeypatch):
    fsq._destination_cache.clear()
    monkeypatch.setattr(fsq, "llm_json", lambda s, u: {"places": [{"name": "Siena", "lat": 43.32, "lng": 11.33, "activity_scores": {"culture": 0.9}}]})
    out = fsq.search_destinations.invoke({"region": "Tuscany, Italy"})
    assert out[0]["name"] == "Siena" and set(out[0]["activity_scores"]) >= {"culture", "nature", "food"}
    fsq._destination_cache.clear()


def test_search_attractions_returns_scheduler_ready_rows(monkeypatch):
    fsq._attraction_cache.clear()
    monkeypatch.setattr(fsq, "get_llm", lambda: object())
    monkeypatch.setattr(fsq, "llm_decide", lambda *a, **k: {"attractions": [
        {"name": "Amber Fort", "category": "culture", "duration_minutes": 120, "rating": 4.6, "opening_hour": 9, "closing_hour": 17, "cost_inr": 200, "lat": 26.98, "lng": 75.85},
        {"name": "Broken", "category": "weird", "lat": "x"}]})
    out = fsq.search_attractions.invoke({"destination": "Jaipur"})
    fort = next(a for a in out if a["name"] == "Amber Fort")
    assert fort["cost_inr"] == 200 and fort["opening_hour"] == 9 and fort["coordinates"]["lat"] == pytest.approx(26.98)
    assert all(a["category"] in {"nature", "adventure", "food", "nightlife", "relaxation", "culture", "shopping", "general"} for a in out)
    fsq._attraction_cache.clear()


# --- hotels, public transport (LLM) -------------------------------------------------------------------------------------

def test_hotels_and_stay_totals(monkeypatch):
    market._hotel_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda s, u: {"hotels": [
        {"name": "Hotel Avenida", "area": "Centro", "price_per_night_inr": 5200, "rating": 4.4},
        {"name": "Casa Simples", "price_per_night_inr": 2100, "rating": 3.9},
        {"name": "Bad Row", "price_per_night_inr": 0}]})
    hotels = market.search_hotels.invoke({"destination": "Lisbon", "budget_tier": "budget"})
    assert [h["price_per_night_inr"] for h in hotels] == [2100, 5200]  # bad row dropped, budget sorts cheapest first
    market._hotel_cache.clear()


def test_public_transport_quote_shape(monkeypatch):
    market._transport_cache.clear()
    monkeypatch.setattr(market, "llm_json", lambda s, u: {"available": True, "operator": "IndiGo", "duration_hours": 3.4, "price_inr": 6500, "note": "via Kochi"})
    out = market.search_public_transport.invoke({"origin": "Delhi", "destination": "Munnar", "mode": "air", "date": "2099-01-05"})
    assert out["available"] and out["mode"] == "air" and out["cost_inr"] == 6500 and "via Kochi" in out["summary"]
    monkeypatch.setattr(market, "llm_json", lambda s, u: {"available": True, "duration_hours": 0, "price_inr": 0})
    assert market.search_public_transport.invoke({"origin": "A", "destination": "B", "mode": "rail"})["available"] is False
    market._transport_cache.clear()


# --- weather ----------------------------------------------------------------------------------------------------------------

def _day(date, condition="Clear", rain_mm=0.0, chance=10, rainy=False):
    return {"date": date, "tmin": 20.0, "tmax": 30.0, "rain_mm": rain_mm, "rain_chance": chance, "condition": condition, "rainy": rainy}


def test_weather_uses_the_forecast_when_close_and_last_year_when_far(monkeypatch):
    from datetime import date, timedelta

    calls = []

    def fake(lat, lng, start, end, historical):
        calls.append(historical)
        n = (end - start).days + 1
        return [_day((start + timedelta(days=i)).isoformat(), "Rain", 12.0, None if historical else 90, True) for i in range(n)]

    monkeypatch.setattr(weather, "_fetch", fake)
    near = date.today() + timedelta(days=3)
    out = weather.trip_weather("Munnar", near.isoformat(), 2)
    assert out["source"] == "forecast" and calls == [False] and len(out["rainy_dates"]) == 2 and "rain likely on 2 of 2" in out["summary"]
    far = date.today() + timedelta(days=60)
    out = weather.trip_weather("Munnar", far.isoformat(), 2)
    assert out["source"] == "last year" and calls[-1] is True and "last year" in out["summary"]
    assert [d["date"] for d in out["days"]] == [far.isoformat(), (far + timedelta(days=1)).isoformat()]  # dated for this year


def test_weather_never_breaks_a_plan(monkeypatch):
    monkeypatch.setattr(weather, "_fetch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    out = weather.trip_weather("Munnar", "2099-01-05", 3)
    assert out["source"] == "unavailable" and out["days"] == []
    assert weather.trip_weather("zzz nowhere zzz", "", 1)["source"] == "unavailable"


def test_rain_and_storms_are_reported(monkeypatch):
    monkeypatch.setattr(weather, "_fetch", lambda lat, lng, s, e, h: [_day(s.isoformat(), "Thunderstorm", 30.0, 95, True)])
    out = weather.check_weather_disruptions.invoke({"destinations": ["Munnar"], "start_date": "2099-01-05", "days": 1})
    assert out["issues"] and "storms on" in out["issues"][0]["message"]
    assert market.check_transport_disruptions.invoke({"destination": "Munnar", "start_date": "2099-01-05"})["disrupted"] is True
    monkeypatch.setattr(weather, "_fetch", lambda lat, lng, s, e, h: [_day(s.isoformat())])
    assert weather.check_weather_disruptions.invoke({"destinations": ["Munnar"], "start_date": "2099-01-05"})["issues"] == []
    assert market.check_transport_disruptions.invoke({"destination": "Munnar", "start_date": "2099-01-05"})["disrupted"] is False


# --- pure tools --------------------------------------------------------------------------------------------------------------

def test_preference_score():
    from tools.preference_scorer import score_preference_match

    hi = score_preference_match.invoke({"preferences": {"nature": 1.0}, "category_scores": {"nature": 0.9}})["score"]
    lo = score_preference_match.invoke({"preferences": {"nature": 1.0}, "category_scores": {"shopping": 0.9}})["score"]
    assert 0 <= lo < hi <= 1


def test_budget_tools():
    from tools.budget_validator import generate_tradeoff_options, validate_budget

    assert validate_budget.invoke({"total_inr": 100, "ceiling_inr": 90}) == {"within_budget": False, "over_by_inr": 10.0}
    assert validate_budget.invoke({"total_inr": 80, "ceiling_inr": 90})["within_budget"] is True
    options = generate_tradeoff_options.invoke({"over_by_inr": 500, "hotel_price_diff_inr": 1000, "activity_cost_inr": 200})["suggestions"]
    assert len(options) == 3 and "hotel" in options[0].lower()


def test_road_costs_depend_on_the_vehicle_and_the_group():
    import tools.cost_calculator as cost

    def price(km, people, vehicle):
        return cost.estimate_road_cost.invoke({"distance_km": km, "travellers": people, "vehicle": vehicle})

    assert price(500, 2, "own_car")["cost_inr"] == 500 * 8, "one car: fuel and tolls"
    assert price(500, 6, "own_car")["vehicles"] == 2 and price(500, 6, "own_car")["cost_inr"] == 500 * 8 * 2, "six people need two cars"
    assert price(500, 6, "tempo_traveller")["vehicles"] == 1 and price(500, 6, "tempo_traveller")["cost_inr"] == 500 * 25
    assert price(500, 13, "tempo_traveller")["vehicles"] == 2
    assert price(100, 10, "bus")["cost_inr"] == 100 * 2.5 * 10, "a bus seat is priced per person"
    assert price(100, 2, "spaceship")["vehicle"] == "taxi", "an unknown vehicle is priced as a taxi"
    assert price(1200, 2, "taxi")["cost_inr"] > price(1200, 2, "own_car")["cost_inr"]


def test_local_transport_is_a_daily_allowance_per_group_of_four():
    import tools.cost_calculator as cost

    assert cost.estimate_local_transport.invoke({"duration_days": 4, "travellers": 2})["total_inr"] == 500 * 1 * 4
    assert cost.estimate_local_transport.invoke({"duration_days": 4, "travellers": 6})["total_inr"] == 500 * 2 * 4


def test_cost_tools(monkeypatch):
    import tools.cost_calculator as cost

    assert cost.calculate_activity_costs.invoke({"activity_costs_inr": [100, 50], "travellers": 3})["total_inr"] == 450
    assert cost.calculate_route_cost.invoke({"leg_costs_inr": [100, 250.5], "travellers": 4})["total_inr"] == 350.5
    cost._food_cache.clear()
    monkeypatch.setattr(cost, "llm_json", lambda s, u: {"per_person_per_day_inr": 1500})
    food = cost.estimate_food_costs.invoke({"duration_days": 4, "travellers": 2, "tier": "mid", "region": "Goa, India"})
    assert food["total_inr"] == 1500 * 4 * 2
    cost._food_cache.clear()


def test_schedule_tools():
    from tools.schedule_validator import check_schedule_conflicts, validate_time_windows

    items = [{"start_hour": 9, "end_hour": 12, "activity_name": "A"}, {"start_hour": 11, "end_hour": 13, "activity_name": "B"}]
    assert validate_time_windows.invoke({"scheduled_items": items})["valid"] is False
    day = {"day_number": 1, "items": [{"start_hour": 6, "end_hour": 12, "activity_name": "A"}, {"start_hour": 12, "end_hour": 21, "activity_name": "B"}]}
    assert any("overpacked" in i["message"] for i in check_schedule_conflicts.invoke({"days": [day]})["issues"])


def test_trip_schema_validation():
    from agents.trip_analyst import validate_trip_schema

    assert validate_trip_schema.invoke({"spec": {"destination_region": "Goa", "duration_days": 4, "budget_inr": 30000}})["valid"] is True
    assert validate_trip_schema.invoke({"spec": {"duration_days": 0}})["valid"] is False


def test_edit_router_directive_accepts_stringly_typed_numbers():
    """A model that emits {"free_days": ["2"]} or a decimal string like "2.5" for a day number
    used to crash with an uncaught ValueError (int("2.5") fails even though the field was
    already confirmed numeric by _num). It must parse, not raise."""
    from agents.edit_router import _directive_from_json

    directive = _directive_from_json({
        "intent": "modify",
        "route_to": "itinerary_architect",
        "lock_updates": {
            "free_days": ["2", "3.0"],
            "light_days": ["4.5"],
            "pinned_activities": {"Museum": "1.0"},
            "day_start_hour": "8.5",
        },
    })
    assert directive is not None
    assert directive.lock_updates.free_days == [2, 3]
    assert directive.lock_updates.light_days == [4]
    assert directive.lock_updates.pinned_activities == {"Museum": 1}
    assert directive.lock_updates.day_start_hour == 8.5


def test_edit_router_tools_read_the_plan(planning_state):
    from agents.destination_agent import destination_agent_node
    from agents.edit_router import _make_tools
    from agents.itinerary_architect import itinerary_architect_node
    from agents.budget_agent import budget_agent_node
    from agents.mobility_agent import mobility_agent_node

    state = {**planning_state, **destination_agent_node(planning_state)}
    for node in (mobility_agent_node, budget_agent_node):
        state = {**state, **node(state)}
    state = {**state, **itinerary_architect_node(state)}
    by = {t.name: t for t in _make_tools(state)}
    day = by["get_plan_day"].invoke({"day_number": 2})
    assert day["found"] and day["city"] and by["get_plan_day"].invoke({"day_number": 99})["found"] is False
    city = state["selected_destinations"][0].name
    assert city in by["find_in_plan"].invoke({"query": city[:4]})["cities"]
    assert isinstance(by["list_alternative_cities"].invoke({})["cities"], list)


# ====================================================================================================
# Provider chain (Groq, then Gemini)
# ====================================================================================================

class FakeModel:
    def __init__(self, name, replies):
        self.name, self.replies, self.calls, self.bound = name, list(replies), 0, None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def bind(self, **kwargs):  # JSON mode
        return self

    def invoke(self, messages):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else {"who": self.name}
        if isinstance(reply, Exception):
            raise reply
        return AIMessage(content=json.dumps(reply))


@pytest.fixture
def providers(monkeypatch):
    llm._cooldown_until.clear()
    monkeypatch.setattr(llm, "LLM_DISABLED", False)

    def install(groq, gemini):
        monkeypatch.setattr(llm, "_providers", lambda: [("groq", groq), ("gemini", gemini)])
        monkeypatch.setitem(llm._RUNNERS, "gemini", llm._react_loop)  # the fakes speak LangChain

    yield install
    llm._cooldown_until.clear()


def test_groq_answers_first(providers):
    groq, gemini = FakeModel("groq", []), FakeModel("gemini", [])
    providers(groq, gemini)
    assert llm.llm_decide(object(), [], "s", "u", max_rounds=1)["who"] == "groq" and gemini.calls == 0


def test_rate_limited_groq_falls_back_to_gemini_and_is_skipped_meanwhile(providers):
    groq = FakeModel("groq", [Exception("Error code: 429 - Rate limit reached. Please try again in 30s.")])
    gemini = FakeModel("gemini", [])
    providers(groq, gemini)
    assert llm.llm_decide(object(), [], "s", "u", max_rounds=1)["who"] == "gemini"
    calls = groq.calls
    assert llm.llm_decide(object(), [], "s", "u", max_rounds=1)["who"] == "gemini" and groq.calls == calls


def test_a_short_wait_is_better_than_giving_up(providers):
    groq, gemini = FakeModel("groq", []), FakeModel("gemini", [])
    providers(groq, gemini)
    llm._cooldown_until["groq"] = time.monotonic() + 0.3
    llm._cooldown_until["gemini"] = time.monotonic() + 0.3
    assert llm.llm_decide(object(), [], "s", "u", max_rounds=1)["who"] == "groq"


def test_a_rate_limited_gemini_key_hands_over_to_the_next_key(monkeypatch):
    llm._cooldown_until.clear()
    monkeypatch.setattr(llm, "LLM_DISABLED", False)
    first = FakeModel("key1", [Exception("429 RESOURCE_EXHAUSTED retry in 30s")])
    second = FakeModel("key2", [])
    monkeypatch.setattr(llm, "_providers", lambda: [("gemini", first), ("gemini-2", second)])
    monkeypatch.setitem(llm._RUNNERS, "gemini", llm._react_loop)  # "gemini-2" runs the Gemini loop too
    assert llm.llm_decide(object(), [], "s", "u", max_rounds=1)["who"] == "key2"
    assert llm._cooldown_left("gemini") > 0 and llm._cooldown_left("gemini-2") == 0
    llm._cooldown_until.clear()


def test_gemini_gets_the_tools_too(providers):
    from tools.budget_validator import validate_budget

    groq = FakeModel("groq", [Exception("429 try again in 30s")])
    gemini = FakeModel("gemini", [])
    providers(groq, gemini)
    llm.llm_decide(object(), [validate_budget], "s", "u")
    assert gemini.bound == [validate_budget]


def test_the_last_round_has_no_tools_so_the_model_must_answer(providers):
    """A model that keeps asking for tools used to run out of rounds and return nothing."""
    from langchain_core.messages import AIMessage
    from tools.budget_validator import validate_budget

    class Greedy(FakeModel):
        def invoke(self, messages):
            self.calls += 1
            if self.bound_now:  # still has tools: ask for another call
                return AIMessage(content="", tool_calls=[{"name": "validate_budget", "args": {"total_inr": 1, "ceiling_inr": 2}, "id": f"c{self.calls}"}])
            return AIMessage(content=json.dumps({"who": "final"}))

        def bind_tools(self, tools):
            clone = Greedy(self.name, [])
            clone.bound_now, clone.calls = True, self.calls
            return clone

        def bind(self, **kw):
            clone = Greedy(self.name, [])
            clone.bound_now = False
            return clone

    groq = Greedy("groq", [])
    groq.bound_now = True
    providers(groq, FakeModel("gemini", []))
    assert llm.llm_decide(object(), [validate_budget], "s", "u", max_rounds=3)["who"] == "final"


def test_a_malformed_tool_call_is_retried_without_tools(providers):
    from tools.budget_validator import validate_budget

    groq = FakeModel("groq", [Exception("400 tool_use_failed: Failed to call a function"), {"who": "groq-json"}])
    providers(groq, FakeModel("gemini", []))
    assert llm.llm_decide(object(), [validate_budget], "s", "u")["who"] == "groq-json"


def test_an_unparseable_reply_fails_over_instead_of_being_accepted(providers):
    """A reply that isn't valid JSON used to still be accepted, because tagging it with
    `_tool_calls` made the otherwise-empty decision dict truthy. That silently defeated
    failover: the next provider was never tried, and callers saw an empty decision as if
    the model had genuinely answered "nothing". It must now fail over."""
    groq = FakeModel("groq", ["not actually json"])  # -> parse_json_blob returns {}
    gemini = FakeModel("gemini", [{"who": "gemini"}])
    providers(groq, gemini)
    assert llm.llm_decide(object(), [], "s", "u", max_rounds=1)["who"] == "gemini"
    assert groq.calls == 1 and gemini.calls == 1


def test_no_provider_configured(monkeypatch):
    monkeypatch.setattr(llm, "LLM_DISABLED", False)
    monkeypatch.setattr(llm, "_providers", lambda: [])
    assert llm.llm_decide(object(), [], "s", "u") == {}


@pytest.mark.parametrize(
    "text,expected",
    [("Please try again in 6.5s.", 6.5), ("Please try again in 1m2.5s", 62.5), ("Please retry in 21.66s.", 21.66), ("retry_delay { seconds: 21 }", 21.0), ("'retryDelay': '34s'", 34.0), ("plain 429", 20.0)],
)
def test_wait_time_is_read_from_the_error(text, expected):
    assert llm._retry_after_seconds(Exception(text)) == pytest.approx(expected)
