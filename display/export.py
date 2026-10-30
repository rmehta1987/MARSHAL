"""Bundle the MARSHAL display into one HTML file that works offline.

    python3 -m display.export --out marshal_display.html
    python3 -m display.export --match connect --limit 10 --out connect_four.html

The file carries the page, fonts, images, the saved games and the Results data.
Open it in any browser, for example from a USB stick on the venue's computer.
Live games need the server.
"""

import argparse
import base64
import json
import mimetypes
import os
import re
import sys

from display.games import GameIndex
from display.runs import RunIndex
from display.server import DECRYPTO_RESULTS, DEFAULT_GAMES, STATIC_DIR, results_payload


def data_uri(path):
    ctype = mimetypes.guess_type(path)[0] or ("font/woff2" if path.endswith(".woff2") else "application/octet-stream")
    if path.endswith(".svg"):
        ctype = "image/svg+xml"
    with open(path, "rb") as f:
        return f"data:{ctype};base64,{base64.b64encode(f.read()).decode()}"


def read(name):
    with open(os.path.join(STATIC_DIR, name), encoding="utf-8") as f:
        return f.read()


def build(games_roots=DEFAULT_GAMES, match="", limit=0, decrypto_dir=DECRYPTO_RESULTS, artifact=False):
    index = GameIndex(games_roots)
    records = index.scan()
    if match:
        words = match.lower().split()
        records = [r for r in records if all(
            w in " ".join([r["title"], r["game"], r.get("source") or "", *(p["name"] for p in r["players"])]).lower()
            for w in words)]
    if limit:
        records = records[:limit]
    results = results_payload(RunIndex("."), index, decrypto_dir)
    assets = {f"/static/{n}": data_uri(os.path.join(STATIC_DIR, n)) for n in ("mascot.png", "method_overview.svg")}
    embed = {"games": records, "results": results, "assets": assets}

    css = read("app.css")
    css = re.sub(r"url\((fonts/[^)]+)\)", lambda m: f"url({data_uri(os.path.join(STATIC_DIR, m.group(1)))})", css)
    js = read("app.js")
    html = read("index.html")
    payload = json.dumps(embed, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    for path, uri in assets.items():
        html = html.replace(f'src="{path}"', f'src="{uri}"')
    html = html.replace('<link rel="stylesheet" href="/static/app.css">', f"<style>\n{css}\n</style>")
    html = html.replace('<script src="/static/app.js"></script>',
                        f"<script>window.MARSHAL_EMBED = {payload};</script>\n<script>\n{js}\n</script>")
    if artifact:
        # A claude.ai Artifact page: the host adds the document skeleton, so ship the
        # title, the viewport promise, the styles and the body's content only.
        title = re.search(r"<title>.*?</title>", html, re.S).group(0)
        style = re.search(r"<style>.*?</style>", html, re.S).group(0)
        body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
        html = f'{title}\n<meta name="claude:viewport-fit-sides" content="cover">\n{style}\n{body}'
    return html, len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="marshal_display.html")
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES, help="Folders with *.marshal-game.json files.")
    parser.add_argument("--match", default="", help="Keep games whose game, model or tag match these words.")
    parser.add_argument("--limit", type=int, default=0, help="Keep at most this many games (0 = all).")
    parser.add_argument("--artifact", action="store_true",
                        help="Write a claude.ai Artifact page (no <html>/<head>/<body> of its own).")
    args = parser.parse_args(argv)
    html, n = build(args.games, args.match, args.limit, artifact=args.artifact)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {args.out}: {n} games, {len(html) / 1e6:.1f} MB. Open it in any browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
