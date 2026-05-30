import math
from pathlib import Path

def generate_stations(
    n: int,
    center_x: float,
    center_z: float,
    radius: float,
    start_angle_deg: float = 0.0,
    clockwise: bool = False,
    station_code: str = "AA",
    filename: str = "STATIONS",
):
    """
    Generate N receivers on a circle and save to an ASCII file.

    Columns:
      1) Station ID (S0001, S0002, ...)
      2) Station code (e.g., 'AA')
      3) X (third column)
      4) Z (fourth column)
      5) 0.0
      6) 0.0
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if not station_code or len(station_code.strip()) == 0:
        raise ValueError("station_code must be a non-empty string")

    step = 360.0 / n
    sign = -1.0 if clockwise else 1.0

    lines = []
    for i in range(n):
        angle_deg = start_angle_deg + sign * i * step
        ang = math.radians(angle_deg)
        x = center_x + radius * math.cos(ang)
        z = center_z + radius * math.sin(ang)

        sid = f"S{i+1:04d}"

       
        line = (
            f"{sid:<6}"         # 'S0001' left-aligned in 6 chars
            f"    {station_code:<2}"
            f"          {x:13.7f}"
            f"         {z:13.7f}"
            f"       0.0"
            f"         0.0"
        )
        lines.append(line)

    Path(filename).write_text("\n".join(lines) + "\n", encoding="ascii")


# --- Example usage ---
if __name__ == "__main__":
    # Edit these values as needed:
    N = 16
    CX, CZ = 500.0, 500.0
    R = 400.0
    START_ANGLE_DEG = 0.0    # 0° on +X axis; increase counterclockwise by default
    CLOCKWISE = False
    CODE = "AA"

    generate_stations(
        n=N,
        center_x=CX,
        center_z=CZ,
        radius=R,
        start_angle_deg=START_ANGLE_DEG,
        clockwise=CLOCKWISE,
        station_code=CODE,
        filename="STATIONS",
    )
    print(f"Wrote {N} stations to 'STATIONS'")
