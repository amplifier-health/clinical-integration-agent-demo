#!/usr/bin/env python3
"""Run the real clinical-integration agent over a scenario.

Two modes:
  (default) print every emitted event as JSONL to stdout.
  --serve   run an SSE server so the Abridge-style UI (mock_agent/viewer.html)
            subscribes and watches the REAL agent reason live.

The agent replays the scenario's precomputed biomarker/transcript inputs and calls
Claude for the reasoning + a real ClinicalTrials.gov lookup — see clinical_agent.py.

Usage:
    python agent/run_agent.py --scenario mock_agent/scenario_womenshealth.json --speed 8
    python agent/run_agent.py --scenario mock_agent/local/scenario_womenshealth_real.json --serve --port 8788
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clinical_agent  # noqa: E402

def load_env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    for path in (os.path.join(HERE, "..", ".env"), ".env"):
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY") and "=" in line:
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            return

def run_stdout(scenario, speed):
    agent = clinical_agent.ClinicalAgent(emit=lambda e: print(json.dumps(e), flush=True))
    agent.run(scenario, speed=speed)

def serve(scenario, speed, port):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs
    viewer = os.path.join(HERE, "..", "mock_agent", "viewer.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/chart":
                self._json({"chart": scenario["chart"], "scenario": scenario["scenario"]})
            elif u.path == "/events":
                sp = float(parse_qs(u.query).get("speed", [speed])[0])
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                def emit(ev):
                    try:
                        self.wfile.write(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        raise SystemExit
                try:
                    clinical_agent.ClinicalAgent(emit=emit).run(scenario, speed=sp)
                except SystemExit:
                    return
            elif u.path in ("/", "/viewer.html"):
                self._send(200, "text/html", open(viewer, "rb").read())
            else:
                self._send(404, "text/plain", "not found")
        def _json(self, o): self._send(200, "application/json", json.dumps(o))
        def _send(self, c, t, b):
            self.send_response(c); self.send_header("Content-Type", t)
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(b if isinstance(b, bytes) else b.encode())

    print(f"live agent server: http://localhost:{port}/  (real Claude reasoning; open the viewer and Play)")
    ThreadingHTTPServer(("", port), H).serve_forever()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--speed", type=float, default=8.0)
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8788)
    a = p.parse_args()
    load_env()
    scenario = json.load(open(a.scenario))
    (serve(scenario, a.speed, a.port) if a.serve else run_stdout(scenario, a.speed))

if __name__ == "__main__":
    main()
