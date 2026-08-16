"""Local browser GUI for tuning straight-replay drag and driveline efficiency."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYZER = ROOT / "analysis/accel/analyze_acceleration.py"
OUTPUT_DIR = ROOT / "analysis/accel/tuning_output"
DEFAULT_DRAG_COEFFICIENT = 1.228784792939579
DEFAULT_MOTOR_TO_WHEEL_EFFICIENCY = 0.8093392555676976
PLOT_NAMES = tuple(f"straight_{number:02d}_comparison.png" for number in range(1, 6))
RUN_LOCK = threading.Lock()

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Straight Replay Tuner</title>
  <style>
    :root { color-scheme: dark; --bg:#101318; --panel:#191e26; --line:#303846; --text:#edf2f7; --muted:#9ba8b8; --accent:#ff8a3d; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,Segoe UI,sans-serif; }
    header { position:sticky; top:0; z-index:2; padding:18px 24px; background:rgba(16,19,24,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(8px); }
    h1 { margin:0 0 14px; font-size:22px; }
    .controls { display:grid; grid-template-columns:minmax(280px,1fr) minmax(280px,1fr) auto; gap:22px; align-items:end; }
    .control label { display:flex; justify-content:space-between; color:var(--muted); margin-bottom:7px; }
    .input-row { display:grid; grid-template-columns:1fr 82px; gap:10px; }
    input { accent-color:var(--accent); }
    input[type=number] { width:100%; padding:7px; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; }
    button { padding:10px 18px; border:0; border-radius:7px; background:var(--accent); color:#17110d; font-weight:700; cursor:pointer; }
    button:disabled { opacity:.45; cursor:wait; }
    #status { margin-top:10px; min-height:22px; color:var(--muted); }
    main { max-width:1500px; margin:auto; padding:22px; display:grid; gap:22px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
    .card h2 { margin:0; padding:12px 16px; font-size:17px; border-bottom:1px solid var(--line); }
    .card img { display:block; width:100%; min-height:260px; object-fit:contain; background:white; }
    @media(max-width:800px) { .controls { grid-template-columns:1fr; } header { position:static; } }
  </style>
</head>
<body>
<header>
  <h1>GNSS/IMU Straight Replay Tuner</h1>
  <div class="controls">
    <div class="control">
      <label><span>Aerodynamic drag coefficient, Cd</span></label>
      <div class="input-row"><input id="dragSlider" type="range" min="0.5" max="5.0" step="0.01"><input id="dragNumber" type="number" min="0" max="5" step="0.001"></div>
    </div>
    <div class="control">
      <label><span>Motor-to-wheel efficiency</span></label>
      <div class="input-row"><input id="effSlider" type="range" min="0.50" max="1.00" step="0.005"><input id="effNumber" type="number" min="0.01" max="1" step="0.001"></div>
    </div>
    <button id="run">Run replay</button>
  </div>
  <div id="status">Ready.</div>
</header>
<main id="plots"></main>
<script>
const defaults={drag:1.228784792939579,efficiency:0.8093392555676976};
const ids=['drag','eff'];
for(const id of ids){
  const slider=document.getElementById(id+'Slider'), number=document.getElementById(id+'Number');
  const value=id==='drag'?defaults.drag:defaults.efficiency; slider.value=value; number.value=value.toFixed(3);
  slider.addEventListener('input',()=>number.value=Number(slider.value).toFixed(3));
  number.addEventListener('input',()=>slider.value=number.value);
}
const plots=document.getElementById('plots');
for(let n=1;n<=5;n++){
  const nn=String(n).padStart(2,'0');
  plots.insertAdjacentHTML('beforeend',`<section class="card"><h2>Straight ${n}</h2><img id="plot${n}" alt="Straight ${n} comparison"></section>`);
}
async function run(){
  const button=document.getElementById('run'), status=document.getElementById('status');
  const dragValue=Number(document.getElementById('dragNumber').value);
  const efficiencyValue=Number(document.getElementById('effNumber').value);
  button.disabled=true; status.textContent='Running five distance-domain replays…';
  try{
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({drag:dragValue,efficiency:efficiencyValue})});
    const result=await response.json(); if(!response.ok) throw new Error(result.error||'Replay failed');
    const stamp=Date.now(); for(let n=1;n<=5;n++){const nn=String(n).padStart(2,'0');document.getElementById('plot'+n).src=`/plots/straight_${nn}_comparison.png?v=${stamp}`;}
    status.textContent=`Done in ${result.elapsed_s.toFixed(1)} s · mean speed RMSE ${result.mean_speed_rmse_mps.toFixed(3)} m/s · mean acceleration RMSE ${result.mean_accel_rmse_mps2.toFixed(3)} m/s²`;
  }catch(error){ status.textContent='Error: '+error.message; }
  finally{ button.disabled=false; }
}
document.getElementById('run').addEventListener('click',run);
run();
</script>
</body>
</html>"""


def run_analysis(drag: float, efficiency: float) -> dict[str, float]:
    if not 0.0 <= drag <= 5.0:
        raise ValueError("drag coefficient must be between 0 and 5")
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(
            "motor-to-wheel efficiency must be greater than 0 and at most 1"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ANALYZER),
        "--output-dir",
        str(OUTPUT_DIR),
        "--drag-coefficient",
        str(drag),
        "--motor-to-wheel-efficiency",
        str(efficiency),
        "--negative-torque-policy",
        "clip",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        creationflags=creationflags,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    report = json.loads(
        (OUTPUT_DIR / "straight_acceleration_metrics.json").read_text(encoding="utf-8")
    )
    metrics = report["metrics"]
    return {
        "mean_speed_rmse_mps": sum(row["speed_rmse_mps"] for row in metrics)
        / len(metrics),
        "mean_accel_rmse_mps2": sum(row["accel_rmse_mps2"] for row in metrics)
        / len(metrics),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(HTML.encode(), "text/html; charset=utf-8")
            return
        name = self.path.split("?", 1)[0].removeprefix("/plots/")
        if self.path.startswith("/plots/") and name in PLOT_NAMES:
            path = OUTPUT_DIR / name
            if path.is_file():
                self._send(path.read_bytes(), "image/png")
                return
        self._send(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self._send(
                b'{"error":"Not found"}', "application/json", HTTPStatus.NOT_FOUND
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                raise ValueError("request is too large")
            request = json.loads(self.rfile.read(length))
            import time

            started = time.perf_counter()
            with RUN_LOCK:
                result = run_analysis(
                    float(request["drag"]), float(request["efficiency"])
                )
            result["elapsed_s"] = time.perf_counter() - started
            self._send(json.dumps(result).encode(), "application/json")
        except Exception as error:  # keep the local UI responsive with useful details
            self._send(
                json.dumps({"error": str(error)}).encode(),
                "application/json",
                HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"GUI: {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Straight replay tuner: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
