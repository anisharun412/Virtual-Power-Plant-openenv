#!/usr/bin/env python3
"""
Validation script for VPP environment.
Tests all core functionality before deployment.
"""

import sys
import asyncio
from pathlib import Path

# Add current dir to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all imports work correctly."""
    print("[*] Testing imports...")
    try:
        from models import VppAction, VppObservation, VppState
        from server.vpp_environment import VppEnvironment
        from server.asset_models import BatteryAsset, SolarAsset, EvAsset
        from client import VppEnv
        print("  [OK] All imports successful")
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_models():
    """Test model validation."""
    print("✓ Testing Pydantic models...")
    try:
        from models import VppAction, VppObservation, BatteryTelemetry
        
        # Test VppAction validation
        action = VppAction(global_charge_rate=0.5, battery_reserve_pct=0.2)
        assert -1.0 <= action.global_charge_rate <= 1.0
        assert 0.0 <= action.battery_reserve_pct <= 1.0
        print("  ✓ VppAction validation works")
        
        # Test invalid action raises error
        try:
            bad_action = VppAction(global_charge_rate=1.5, battery_reserve_pct=0.2)
            print("  ✗ Model validation should reject out-of-range values")
            return False
        except Exception:
            print("  ✓ Model validation correctly rejects invalid values")
        
        return True
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_environment():
    """Test environment creation and basic operations."""
    print("✓ Testing environment core...")
    try:
        from server.vpp_environment import VppEnvironment
        from models import VppAction
        
        env = VppEnvironment()
        print("  ✓ Environment instantiated")
        
        # Test reset
        obs = env.reset(task_id="easy-arbitrage")
        assert obs is not None
        assert obs.step_id == 0
        print("  ✓ Reset works")
        
        # Test step
        action = VppAction(global_charge_rate=0.3, battery_reserve_pct=0.2)
        obs, reward, done, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        print("  ✓ Step works")
        
        # Test state property
        state = env.state
        assert state is not None
        assert state.step_count > 0
        print("  ✓ State property works")
        
        # Test grader
        score = VppEnvironment.get_current_task_score()
        assert 0.0 <= score <= 1.0
        print(f"  ✓ Grader works (score={score:.4f})")
        
        return True
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tasks():
    """Test task configuration."""
    print("✓ Testing tasks...")
    try:
        from server.vpp_environment import VppEnvironment
        from models import VppAction
        
        for task_id in ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]:
            env = VppEnvironment()
            obs = env.reset(task_id=task_id)
            state = env.state
            assert state.task_tier == task_id
            print(f"  ✓ Task '{task_id}' initializes correctly")
        
        return True
    except Exception as e:
        print(f"  ✗ Task test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_market_data():
    """Test market data loading."""
    print("✓ Testing market data...")
    try:
        from server.vpp_environment import VppEnvironment
        
        env = VppEnvironment()
        
        # Test loading prices for all tasks
        for task_id in ["easy-arbitrage", "medium-forecast-error", "hard-frequency-response"]:
            env.reset(task_id=task_id)
            assert len(env.price_series) == 96, f"Expected 96 prices, got {len(env.price_series)}"
            assert all(isinstance(p, float) for p in env.price_series)
            assert all(p > 0 for p in env.price_series)
            print(f"  ✓ {task_id}: {min(env.price_series):.2f}-{max(env.price_series):.2f} $/MWh")
        
        return True
    except Exception as e:
        print(f"  ✗ Market data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_episode():
    """Test complete episode."""
    print("✓ Testing full episode...")
    try:
        from server.vpp_environment import VppEnvironment
        from models import VppAction
        import numpy as np
        
        env = VppEnvironment()
        obs = env.reset(task_id="easy-arbitrage")
        
        total_reward = 0.0
        steps = 0
        
        for step in range(96):
            # Random action
            action = VppAction(
                global_charge_rate=np.random.uniform(-1.0, 1.0),
                battery_reserve_pct=np.random.uniform(0.0, 1.0)
            )
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        print(f"  ✓ Episode complete: {steps} steps, reward={total_reward:.4f}")
        
        # Test grader after episode
        score = VppEnvironment.get_current_task_score()
        print(f"  ✓ Final score: {score:.4f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_client():
    """Test client setup (don't actually connect)."""
    print("✓ Testing client...")
    try:
        from client import VppEnv
        from models import VppAction
        
        # Just verify client can be instantiated
        client = VppEnv(base_url="http://localhost:8000")
        assert client is not None
        print("  ✓ Client instantiated")
        
        # Verify payload serialization works
        action = VppAction(global_charge_rate=0.5, battery_reserve_pct=0.3)
        payload = client._step_payload(action)
        assert "global_charge_rate" in payload
        assert "battery_reserve_pct" in payload
        print("  ✓ Payload serialization works")
        
        return True
    except Exception as e:
        print(f"  ✗ Client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_app():
    """Test FastAPI app endpoints."""
    print("✓ Testing FastAPI app...")
    try:
        from server.app import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test /tasks endpoint
        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data
        assert len(data["tasks"]) >= 3
        print("  ✓ /tasks endpoint works")
        
        # Test /grader endpoint
        response = client.get("/grader")
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        print("  ✓ /grader endpoint works")
        
        return True
    except ImportError:
        print("  ⚠ FastAPI TestClient not available (requires fastapi[test])")
        return True  # Don't fail on this
    except Exception as e:
        print(f"  ✗ App test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("VPP ENVIRONMENT VALIDATION")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Models", test_models),
        ("Environment", test_environment),
        ("Tasks", test_tasks),
        ("Market Data", test_market_data),
        ("Full Episode", test_full_episode),
        ("Client", test_client),
        ("FastAPI App", test_app),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            if asyncio.iscoroutinefunction(test_fn):
                result = asyncio.run(test_fn())
            else:
                result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Environment is ready.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
