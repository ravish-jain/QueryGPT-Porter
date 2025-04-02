import pytest
from core.yaml_schema_parser import SchemaLoader
from core.few_shot_examples_loader import ExampleLoader
from app import generate_sql

@pytest.fixture
def test_schema():
    return SchemaLoader("tests/data/schema.yml")

@pytest.fixture 
def test_examples():
    return ExampleLoader("tests/data/examples.json")

def test_schema_loading(test_schema):
    assert "customers" in test_schema.tables
    assert len(test_schema.tables["partner_onboarding_lead_creation"].columns) > 0

def test_example_retrieval(test_examples):
    examples = test_examples.get_relevant_examples("Count of leads and drivers created in Feb of 2025", k=2)
    assert len(examples) == 2
    assert "partner_onboarding_lead_id" in examples[0]["sql"].lower()
    assert "driver_id" in examples[0]["sql"].lower()

def test_sql_generation(monkeypatch, test_schema, test_examples):
    # Temporarily replace the global schema_loader and example_loader
    import app
    monkeypatch.setattr(app, "schema_loader", test_schema)
    monkeypatch.setattr(app, "example_loader", test_examples)
    
    result = generate_sql("Write a query to get the name of the city with most drivers created in 2025")
    assert "SELECT" in result["sql"]
    assert "2025" in result["sql"]
    assert "city_name" in result["sql"].lower()