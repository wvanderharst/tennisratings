"""One-off probe of results.tennisdata.com: robots.txt, and which data requests one match page makes (no bulk download)."""
import asyncio, json, sys, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "https://results.tennisdata.com/en/results/ecef0cbb-a99e-49ad-88a3-e529b18065ee"
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

def get(u):
    try:
        with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": UA}), timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception as e:
        return getattr(e, "code", None), str(e)

for path in ("/robots.txt",):
    st, body = get("https://results.tennisdata.com" + path); print(f"==== {path} -> {st}\n{body[:3000]}\n")
st, html = get(URL); print(f"==== page (plain GET) -> {st}, {len(html)} chars\n{html[:1500]}\n")

async def main():
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(user_agent=UA); seen = []
        async def on_resp(r):
            ct = r.headers.get("content-type", "")
            if r.request.resource_type in ("xhr", "fetch") or "json" in ct:
                try: body = await r.text()
                except Exception: body = ""
                seen.append((r.status, r.request.method, r.url, ct, len(body), body[:1500]))
        pg.on("response", on_resp)
        await pg.goto(URL, wait_until="networkidle", timeout=60000); await pg.wait_for_timeout(4000)
        txt = await pg.inner_text("body")
        print(f"==== rendered text ({len(txt)} chars)\n{txt[:5000]}\n")
        links = await pg.eval_on_selector_all("a", "as => as.map(a => a.textContent.trim() + ' -> ' + a.href)")
        print("==== links\n" + "\n".join(l for l in links if any(k in l.lower() for k in ("term", "legal", "privacy", "cond", "api", "licen", "copyright")))[:3000])
        print(f"\n==== data requests ({len(seen)})")
        for s in seen: print(f"\n-- {s[0]} {s[1]} {s[2]}\n   {s[3]} {s[4]} bytes\n   {s[5]}")
        await b.close()
asyncio.run(main())
