"""Parse IFSC Speed Route Map from PDF and extract hold coordinates.

This module extracts the official IFSC 15m speed climbing route map from the
IFSC Speed Licence Rules PDF document and converts hold positions to metric
coordinates based on the standard 125mm grid spacing.

References:
    - IFSC Speed Licence Rules: docs/IFSC_Speed_Licence_Rules.pdf
    - Wall specifications: 15m height, 3m width per lane, 5° overhang
    - Panel specifications: 1500mm × 1500mm with 125mm hold spacing
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IFSCRouteMapParser:
    """Parser for IFSC Speed Route Map from PDF."""

    # Grid specifications from IFSC standards
    HOLD_SPACING_MM = 125.0
    PANEL_SIZE_MM = 1500.0
    WALL_HEIGHT_M = 15.0
    WALL_WIDTH_M = 3.0
    OVERHANG_DEGREES = 5.0

    # Grid system: A-M (13 columns), 1-10 (10 rows)
    # A=0, B=1, C=2, ..., M=12
    GRID_LETTERS = 'ABCDEFGHIJKLM'

    def __init__(self, pdf_path: str):
        """Initialize parser.

        Args:
            pdf_path: Path to IFSC_Speed_Licence_Rules.pdf
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    def parse_grid_position(self, position: str) -> Tuple[int, int]:
        """Parse grid position like 'F4' or '4M' to (column, row) indices.

        Args:
            position: Grid position (e.g., 'F4', 'A10', 'M8', '4M', '9G')
                     Format can be either LETTER+NUMBER or NUMBER+LETTER

        Returns:
            Tuple of (column_index, row_index) where both are 0-indexed
            Example: 'F4' -> (5, 3), '4M' -> (12, 3)

        Raises:
            ValueError: If position format is invalid
        """
        position = position.strip().upper()
        if len(position) < 2:
            raise ValueError(f"Invalid position format: {position}")

        # Try letter-first format (e.g., 'F4', 'A10')
        if position[0] in self.GRID_LETTERS:
            letter = position[0]
            number_str = position[1:]
        # Try number-first format (e.g., '4M', '10B')
        elif position[0].isdigit():
            # Find where letter starts
            for i, char in enumerate(position):
                if char in self.GRID_LETTERS:
                    number_str = position[:i]
                    letter = position[i]
                    break
            else:
                raise ValueError(f"No letter found in position: {position}")
        else:
            raise ValueError(f"Invalid position format: {position}")

        if letter not in self.GRID_LETTERS:
            raise ValueError(f"Invalid column letter: {letter}")

        try:
            number = int(number_str)
            if not (1 <= number <= 10):
                raise ValueError(f"Row number must be 1-10, got: {number}")
        except ValueError:
            raise ValueError(f"Invalid row number: {number_str}")

        column_index = self.GRID_LETTERS.index(letter)
        row_index = number - 1  # Convert to 0-indexed

        return column_index, row_index

    def grid_to_panel_coordinates(
        self, column_idx: int, row_idx: int
    ) -> Tuple[float, float]:
        """Convert grid indices to panel-local coordinates in mm.

        Args:
            column_idx: Column index (0-12 for A-M)
            row_idx: Row index (0-9 for 1-10)

        Returns:
            Tuple of (x_mm, y_mm) relative to bottom-left of panel
        """
        # X coordinate: column_idx * hold_spacing
        x_mm = column_idx * self.HOLD_SPACING_MM

        # Y coordinate: row_idx * hold_spacing
        # Row 1 (index 0) is at the bottom
        y_mm = row_idx * self.HOLD_SPACING_MM

        return x_mm, y_mm

    def panel_to_wall_coordinates(
        self, panel_name: str, x_panel_mm: float, y_panel_mm: float
    ) -> Tuple[float, float]:
        """Convert panel-local coordinates to wall coordinates in meters.

        Args:
            panel_name: Panel identifier (e.g., 'DX1', 'SN5', 'SN 8')
                       DX = right lane (Destra), SN = left lane (Sinistra)
                       Number = panel level (1 = bottom, 10 = top)
            x_panel_mm: X coordinate within panel (0-1500mm)
            y_panel_mm: Y coordinate within panel (0-1500mm)

        Returns:
            Tuple of (x_m, y_m) relative to bottom-left corner of wall
        """
        # Parse panel name - normalize by removing spaces and converting to uppercase
        panel_name = panel_name.strip().replace(' ', '').upper()
        if len(panel_name) < 3:
            raise ValueError(f"Invalid panel name: {panel_name}")

        lane = panel_name[:2]  # DX or SN
        try:
            panel_level = int(panel_name[2:])
        except ValueError:
            raise ValueError(f"Invalid panel level in: {panel_name}")

        # Lane offset (DX on right, SN on left)
        # For left lane (SN): x_wall starts from 0
        # For right lane (DX): x_wall starts from wall_width (3m)
        if lane == 'SN':
            x_lane_offset_m = 0.0
        elif lane == 'DX':
            x_lane_offset_m = self.WALL_WIDTH_M
        else:
            raise ValueError(f"Invalid lane identifier: {lane}")

        # Y offset based on panel level
        # Panel 1 is at ground level (0.2m above ground per IFSC rules)
        # Each panel is 1.5m tall
        y_panel_offset_m = (panel_level - 1) * (self.PANEL_SIZE_MM / 1000.0)

        # Convert to wall coordinates
        x_wall_m = x_lane_offset_m + (x_panel_mm / 1000.0)
        y_wall_m = y_panel_offset_m + (y_panel_mm / 1000.0)

        return x_wall_m, y_wall_m

    def extract_route_table(self) -> List[Dict[str, str]]:
        """Extract route table from PDF page 8.

        Returns:
            List of dictionaries with keys: panel, kind_of_hold, position, orientation
        """
        logger.info(f"Extracting route table from {self.pdf_path}")

        with pdfplumber.open(self.pdf_path) as pdf:
            # Route table is on page 8 (index 7)
            if len(pdf.pages) < 8:
                raise ValueError(f"PDF has only {len(pdf.pages)} pages, expected at least 8")

            page = pdf.pages[7]  # Page 8 (0-indexed)
            tables = page.extract_tables()

            if not tables:
                raise ValueError("No tables found on page 8")

            # First table contains the route
            table = tables[0]
            if len(table) < 2:
                raise ValueError("Table is empty or has no data rows")

            # Parse table (skip header row)
            route_data = []
            for row in table[1:]:
                if len(row) < 4:
                    continue  # Skip incomplete rows

                panel, kind, position, orientation = row[:4]

                # Clean up values
                panel = panel.strip() if panel else ""
                kind = kind.strip() if kind else ""
                position = position.strip() if position else ""
                orientation = orientation.strip() if orientation else ""

                if panel and position:  # Only include valid entries
                    route_data.append({
                        "panel": panel,
                        "kind_of_hold": kind,
                        "position": position,
                        "orientation": orientation
                    })

            logger.info(f"Extracted {len(route_data)} holds from route table")
            return route_data

    def parse_route_map(self) -> Dict:
        """Parse complete route map with metric coordinates.

        Returns:
            Dictionary with wall specifications and hold coordinates:
            {
                "wall": {
                    "height_m": 15.0,
                    "width_m": 3.0,
                    "overhang_degrees": 5.0,
                    "hold_spacing_mm": 125
                },
                "holds": [
                    {
                        "hold_num": 1,
                        "panel": "DX1",
                        "kind": "FOOT HOLD",
                        "grid_position": "F4",
                        "grid_orientation": "G4",
                        "grid_x": 5,
                        "grid_y": 3,
                        "panel_x_mm": 625.0,
                        "panel_y_mm": 375.0,
                        "wall_x_m": 3.625,
                        "wall_y_m": 0.375,
                        "description": "Hold 1 on panel DX1"
                    },
                    ...
                ]
            }
        """
        # Extract table data
        route_data = self.extract_route_table()

        # Build output structure
        output = {
            "wall": {
                "height_m": self.WALL_HEIGHT_M,
                "width_m": self.WALL_WIDTH_M,
                "overhang_degrees": self.OVERHANG_DEGREES,
                "hold_spacing_mm": self.HOLD_SPACING_MM
            },
            "holds": []
        }

        # Process each hold
        for i, hold_data in enumerate(route_data, 1):
            try:
                panel = hold_data["panel"]
                position = hold_data["position"]
                orientation = hold_data["orientation"]
                kind = hold_data["kind_of_hold"]

                # Parse grid position
                col_idx, row_idx = self.parse_grid_position(position)

                # Convert to panel coordinates (mm)
                panel_x_mm, panel_y_mm = self.grid_to_panel_coordinates(col_idx, row_idx)

                # Convert to wall coordinates (m)
                wall_x_m, wall_y_m = self.panel_to_wall_coordinates(
                    panel, panel_x_mm, panel_y_mm
                )

                hold_entry = {
                    "hold_num": i,
                    "panel": panel,
                    "kind": kind,
                    "grid_position": position,
                    "grid_orientation": orientation,
                    "grid_x": col_idx,
                    "grid_y": row_idx,
                    "panel_x_mm": panel_x_mm,
                    "panel_y_mm": panel_y_mm,
                    "wall_x_m": round(wall_x_m, 4),
                    "wall_y_m": round(wall_y_m, 4),
                    "description": f"Hold {i} on panel {panel} at position {position}"
                }

                output["holds"].append(hold_entry)

            except Exception as e:
                logger.warning(f"Failed to parse hold {i}: {e}")
                continue

        logger.info(f"Successfully parsed {len(output['holds'])} holds")
        return output

    def save_to_json(self, output_path: str):
        """Parse route map and save to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        route_map = self.parse_route_map()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(route_map, f, indent=2)

        logger.info(f"Route map saved to {output_path}")
        logger.info(f"Total holds: {len(route_map['holds'])}")


def main():
    """Main entry point for route map parser."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse IFSC Speed Route Map from PDF"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="docs/IFSC_Speed_Licence_Rules.pdf",
        help="Path to IFSC PDF"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/ifsc_route_coordinates.json",
        help="Path to output JSON"
    )

    args = parser.parse_args()

    # Parse and save
    route_parser = IFSCRouteMapParser(args.pdf)
    route_parser.save_to_json(args.output)

    print(f"\n✅ Route map extracted successfully!")
    print(f"   Input:  {args.pdf}")
    print(f"   Output: {args.output}")


if __name__ == "__main__":
    main()
