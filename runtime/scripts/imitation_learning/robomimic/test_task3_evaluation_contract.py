from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np


CONTRACT_PATH = Path(__file__).resolve().parents[3] / "pick_place/task3_evaluation_contract.py"
SPEC = importlib.util.spec_from_file_location("task3_evaluation_contract", CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG = CONTRACT.TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
evaluate_task3_carry_basket_success = CONTRACT.evaluate_task3_carry_basket_success


def _payload(*, basket_x: float = 0.0, latched: bool = True, moving: bool = False) -> dict:
    frames = 10
    stool = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), (frames, 1))
    # Basket bottom equals the 0.435 m stool seat.
    basket = np.tile(
        np.asarray([basket_x, 0.0, 0.435, 1.0, 0.0, 0.0, 0.0]),
        (frames, 1),
    )
    velocity = np.zeros((frames, 6), dtype=np.float64)
    if moving:
        velocity[:, 0] = 0.2
    return {
        "object_root_pose": basket,
        "robot_hand_open": np.ones((frames, 2), dtype=np.float64),
        "object_root_velocity": velocity,
        "placement_root_pose": stool,
        "door_joint_position": np.linspace(0.0, math.radians(62.0), frames),
        "door_contact_latched": np.full(frames, latched, dtype=bool),
    }


def test_task3_success_requires_stool_landing_and_door_trigger() -> None:
    result = evaluate_task3_carry_basket_success(
        _payload(), TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
    )
    assert result["ok"]
    assert result["checks"]["door_triggered"]
    assert result["door_closed_angle_reached"]


def test_task3_rejects_outside_or_moving_basket() -> None:
    outside = evaluate_task3_carry_basket_success(
        _payload(basket_x=0.4), TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
    )
    moving = evaluate_task3_carry_basket_success(
        _payload(moving=True), TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
    )
    assert not outside["ok"]
    assert "basket_center_inside_stool_seat" in outside["reasons"]
    assert not moving["ok"]
    assert "linear_speed_settled" in moving["reasons"]


def test_task3_rejects_missing_door_trigger() -> None:
    payload = _payload(latched=False)
    payload["door_joint_position"][:] = 0.0
    result = evaluate_task3_carry_basket_success(
        payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
    )
    assert not result["ok"]
    assert "door_triggered" in result["reasons"]


def test_task3_stool_distribution_is_not_the_walk2_endpoint_distribution() -> None:
    sim_cfg = CONTRACT.TASK3_CARRY_BASKET_SIM_EVALUATION_CFG
    walk2 = sim_cfg["walk2_endpoint"]
    placement = sim_cfg["placement"]
    assert placement["correlated_with_robot_object_scene"] is True
    assert placement["source"].endswith("task3_training_scenarios_386_full_init.jsonl")
    walk2_min_x = walk2["center_xy"][0] - 0.5 * walk2["full_size_xy"][0]
    assert placement["x_range"][0] < walk2_min_x - 1.0


def test_task3_reports_maximum_basket_lift() -> None:
    payload = _payload()
    payload["object_root_pose"][4:, 2] += 0.21
    result = evaluate_task3_carry_basket_success(
        payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG
    )
    assert math.isclose(result["max_object_lift"], 0.21, abs_tol=1.0e-9)


def test_task3_rejects_stationary_basket_above_seat() -> None:
    payload = _payload()
    payload["object_root_pose"][:, 2] += 0.1675468863
    result = evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)
    assert not result["ok"]
    assert "basket_supported_by_stool" in result["reasons"]


def test_task3_requires_both_hands_open() -> None:
    for hand in (0, 1):
        payload = _payload()
        payload["robot_hand_open"][:, hand] = 0.0
        result = evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)
        assert not result["ok"]
        assert "hands_released" in result["reasons"]


def test_task3_rejects_missing_hand_measurements() -> None:
    payload = _payload()
    del payload["robot_hand_open"]
    result = evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)
    assert not result["ok"]
    assert "robot_hand_open" in result["missing"]


def test_task3_requires_stability_throughout_window() -> None:
    for component, speed in ((0, 0.2), (3, 1.0)):
        payload = _payload()
        payload["object_root_velocity"][-2, component] = speed
        result = evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)
        assert not result["ok"]


def test_task3_rejects_release_only_on_last_frame() -> None:
    payload = _payload()
    payload["robot_hand_open"][:-1, :] = 0.0
    assert not evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)["ok"]


def test_task3_requires_entire_window_over_seat() -> None:
    payload = _payload()
    payload["object_root_pose"][0, 0] = 0.4
    assert not evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)["ok"]


def test_task3_requires_full_stability_window() -> None:
    payload = {key: value[:9] for key, value in _payload().items()}
    result = evaluate_task3_carry_basket_success(payload, TASK3_CARRY_BASKET_REPLAY_SUCCESS_CFG)
    assert not result["ok"]
    assert "stable_window_complete" in result["reasons"]
