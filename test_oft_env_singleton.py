#!/usr/bin/env python3
"""Quick test to verify OFT_env is only created once per session"""

import pytest
from OpenFUSIONToolkit import OFT_env

# Global tracker to count how many times fixture is called
fixture_call_count = 0

@pytest.fixture(scope="session")
def oft_env_fixture():
    global fixture_call_count
    fixture_call_count += 1
    print(f"\n[FIXTURE CALLED - Call #{fixture_call_count}]")
    oft_env = OFT_env()
    return oft_env


def test_first_use(oft_env_fixture):
    """First test using the fixture"""
    print(f"\n[TEST 1] Using oft_env_fixture")
    assert oft_env_fixture is not None
    print(f"[TEST 1] Fixture call count: {fixture_call_count}")
    

def test_second_use(oft_env_fixture):
    """Second test using the fixture"""
    print(f"\n[TEST 2] Using oft_env_fixture")
    assert oft_env_fixture is not None
    print(f"[TEST 2] Fixture call count: {fixture_call_count}")


def test_third_use(oft_env_fixture):
    """Third test using the fixture"""
    print(f"\n[TEST 3] Using oft_env_fixture")
    assert oft_env_fixture is not None
    print(f"[TEST 3] Fixture call count: {fixture_call_count}")


def test_same_instance(oft_env_fixture):
    """Verify all tests get the same instance"""
    print(f"\n[TEST 4] Checking instance consistency")
    assert oft_env_fixture is not None
    print(f"[TEST 4] Fixture call count: {fixture_call_count}")
    # If we reach here, the session-scoped fixture worked correctly
    assert fixture_call_count == 1, f"Expected fixture to be called once, but was called {fixture_call_count} times!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
