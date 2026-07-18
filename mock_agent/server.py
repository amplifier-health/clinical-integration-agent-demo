#!/usr/bin/env python3
"""Mock clinical-integration agent — replays a scenario as a live event stream.

Dependency-free (Python stdlib). Serves Server-Sent Events so a browser UI can
subscribe with EventSource and route each typed event to the right Abridge surface.
The real backend swaps this for a WebSocket agent that calls the live API — the UI
contract (the event stream) stays identical.

Endpoints:
  GET /chart                     -> scenario chart JSON (patient header/problem list)
  GET /events?speed=8            -> SSE stream; replays the scenario time-compressed
  GET /                          -> built-in event viewer (mock_agent/viewer.html)

Run:  python mock_agent/server.py [--port 8787] [--scenario scenario_respiratory.json]
"""
import json, os, time, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIO = {}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # quiet

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if body: self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        u = urlparse(self.path)
        # backend-shaped endpoints so the viewer speaks one contract everywhere
        if u.path == "/patients":
            self._send(200, "application/json", json.dumps([{"id": "demo", "name": SCENARIO["chart"].get("name")}]))
        elif u.path.startswith("/patients/") and u.path.endswith("/chart"):
            self._send(200, "application/json", json.dumps(SCENARIO["chart"]))
        elif u.path == "/chart":  # legacy
            self._send(200, "application/json", json.dumps({"chart": SCENARIO["chart"]}))
        elif u.path == "/events":
            self.stream(float(parse_qs(u.query).get("speed", ["8"])[0]))
        elif u.path in ("/", "/viewer.html"):
            self._send(200, "text/html", open(os.path.join(HERE, "viewer.html"), "rb").read())
        else:
            self._send(404, "text/plain", "not found")

    def do_POST(self):  # viewer POSTs /patients/{id}/visits/start; the stream replays on GET /events
        self._send(202, "application/json", json.dumps({"status": "started"}))

    def stream(self, speed):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        events = SCENARIO["events"]
        # announce, then replay on a compressed clock relative to the first event
        t0 = events[0]["t"]
        wall = time.monotonic()
        try:
            self._evt("meta", {"scenario": SCENARIO["scenario"], "speed": speed,
                               "encounter_seconds": SCENARIO["encounter_seconds"]})
            for e in events:
                target = wall + (e["t"] - t0) / speed
                delay = target - time.monotonic()
                if delay > 0: time.sleep(delay)
                self._evt(e["type"], e)
            self._evt("end", {})
        except (BrokenPipeError, ConnectionResetError):
            return

    def _evt(self, type, data):
        # SSE frame: named event + JSON data line
        self.wfile.write(f"event: {type}\ndata: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

def main():
    global SCENARIO
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--scenario", default="scenario_womenshealth.json")
    a = p.parse_args()
    path = a.scenario if os.path.isabs(a.scenario) else os.path.join(HERE, a.scenario)
    SCENARIO = json.load(open(path))
    print(f"mock-agent: {SCENARIO['scenario']} ({len(SCENARIO['events'])} events) on http://localhost:{a.port}")
    print(f"  viewer:  http://localhost:{a.port}/")
    print(f"  stream:  http://localhost:{a.port}/events?speed=8")
    ThreadingHTTPServer(("", a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
