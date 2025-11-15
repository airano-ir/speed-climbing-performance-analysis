"""Tests for calibration module."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration.ifsc_route_map import IFSCRouteMapParser


def test_grid_position_parsing():
    """Test parsing of grid positions in both formats."""
    parser = IFSCRouteMapParser("docs/IFSC_Speed_Licence_Rules.pdf")

    # Test letter-first format
    col, row = parser.parse_grid_position("F4")
    assert col == 5 and row == 3, f"F4 should be (5, 3), got ({col}, {row})"

    col, row = parser.parse_grid_position("A10")
    assert col == 0 and row == 9, f"A10 should be (0, 9), got ({col}, {row})"

    col, row = parser.parse_grid_position("M8")
    assert col == 12 and row == 7, f"M8 should be (12, 7), got ({col}, {row})"

    # Test number-first format
    col, row = parser.parse_grid_position("4M")
    assert col == 12 and row == 3, f"4M should be (12, 3), got ({col}, {row})"

    col, row = parser.parse_grid_position("10B")
    assert col == 1 and row == 9, f"10B should be (1, 9), got ({col}, {row})"

    print("✓ Grid position parsing works correctly")


def test_panel_coordinates():
    """Test conversion to panel coordinates."""
    parser = IFSCRouteMapParser("docs/IFSC_Speed_Licence_Rules.pdf")

    # F4 = column 5, row 3
    x, y = parser.grid_to_panel_coordinates(5, 3)
    assert x == 625.0 and y == 375.0, f"Expected (625.0, 375.0), got ({x}, {y})"

    # A1 = column 0, row 0
    x, y = parser.grid_to_panel_coordinates(0, 0)
    assert x == 0.0 and y == 0.0, f"Expected (0.0, 0.0), got ({x}, {y})"

    # M10 = column 12, row 9
    x, y = parser.grid_to_panel_coordinates(12, 9)
    assert x == 1500.0 and y == 1125.0, f"Expected (1500.0, 1125.0), got ({x}, {y})"

    print("✓ Panel coordinate conversion works correctly")


def test_wall_coordinates():
    """Test conversion to wall coordinates."""
    parser = IFSCRouteMapParser("docs/IFSC_Speed_Licence_Rules.pdf")

    # DX1 (right lane, panel 1), position F4
    x, y = parser.panel_to_wall_coordinates("DX1", 625.0, 375.0)
    assert x == 3.625 and y == 0.375, f"Expected (3.625, 0.375), got ({x}, {y})"

    # SN2 (left lane, panel 2), position G3
    x, y = parser.panel_to_wall_coordinates("SN2", 750.0, 250.0)
    assert x == 0.75 and y == 1.75, f"Expected (0.75, 1.75), got ({x}, {y})"

    print("✓ Wall coordinate conversion works correctly")


def test_route_map_output():
    """Test that route map JSON is valid."""
    config_path = Path("configs/ifsc_route_coordinates.json")

    if not config_path.exists():
        print("⚠ Route map JSON not found, skipping test")
        return

    with open(config_path) as f:
        data = json.load(f)

    # Verify structure
    assert "wall" in data, "Missing 'wall' key"
    assert "holds" in data, "Missing 'holds' key"

    # Verify wall specs
    wall = data["wall"]
    assert wall["height_m"] == 15.0, "Wall height should be 15.0m"
    assert wall["width_m"] == 3.0, "Wall width should be 3.0m"
    assert wall["overhang_degrees"] == 5.0, "Overhang should be 5.0°"
    assert wall["hold_spacing_mm"] == 125.0, "Hold spacing should be 125mm"

    # Verify holds
    holds = data["holds"]
    assert len(holds) > 0, "No holds found"
    print(f"✓ Route map JSON is valid with {len(holds)} holds")

    # Check first hold structure
    hold = holds[0]
    required_keys = [
        "hold_num", "panel", "kind", "grid_position", "grid_orientation",
        "grid_x", "grid_y", "panel_x_mm", "panel_y_mm",
        "wall_x_m", "wall_y_m", "description"
    ]
    for key in required_keys:
        assert key in hold, f"Missing key '{key}' in hold data"

    print("✓ All hold entries have correct structure")


if __name__ == "__main__":
    print("Testing IFSC Route Map Parser...\n")
    test_grid_position_parsing()
    test_panel_coordinates()
    test_wall_coordinates()
    test_route_map_output()
    print("\n✅ All calibration tests passed!")
