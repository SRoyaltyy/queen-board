# Queen board

Same shape as the fullscan `.io` dashboards: a static GitHub Pages front end. Queen itself does not run on Pages. The page plays a legal game in the browser and calls a GPU bridge for analysis.

Fixed URL once Pages is on:

https://sroyaltyy.github.io/queen-board/

## Pages

Repo Settings → Pages → Deploy from branch → `main` / root. Wait a minute. If it 404s, that toggle is the cause.

## Bridge (the part that actually runs Queen)

On the H100 (or other BF16 CUDA 13 box), after the model card install:

```bash
python bridge.py --model-dir ~/queen_pawn-8 --port 8787
```

The dashboard is HTTPS, so the bridge URL must be HTTPS too. A free tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8787
```

Paste the `https://….trycloudflare.com` URL into the dashboard endpoint field. It is stored in this browser only.

`POST /analyze` body: `{"fen": "...", "history": ["prior fen", ...]}`. Response is the infer.py JSON (`text`, `best_move_uci`, `move_source`, `critical_line`).
