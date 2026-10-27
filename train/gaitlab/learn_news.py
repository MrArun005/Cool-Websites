# Add a news item to Bolt Learning Live (newest first): python news.py <kind good|bad|plan|news> <title> <body> [iso time]
import json, sys, datetime as dt, pathlib
p = pathlib.Path(__file__).with_name("learn") / "site" / "data" / "news.json"
N = json.loads(p.read_text()) if p.exists() else dict(items=[])
kind, title, body = sys.argv[1:4]
t = sys.argv[4] if len(sys.argv) > 4 else dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
N["items"].append(dict(t=t, kind=kind, title=title, body=body))
N["items"].sort(key=lambda n: n["t"], reverse=True)
N["updated"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
p.write_text(json.dumps(N, ensure_ascii=False, indent=1)); print(len(N["items"]), "items")
