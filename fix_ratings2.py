import requests
from bs4 import BeautifulSoup
import pandas as pd
import time, random
from fuzzywuzzy import fuzz
import re
# replace existing get_rating_from_search with this version
def get_rating_from_search(product, brand):
    """
    Robust search: pick anchors with /pd/ only, treat each anchor as a candidate,
    extract title from the anchor, find rating inside the anchor's parent card,
    use combined fuzzy matching, require a minimum score, return rating & count.
    """
    base_url = "https://www.bigbasket.com"
    query = f"{product} {brand}".replace(" ", "+")
    search_url = f"{base_url}/ps/?q={query}&nc=as"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/"
    }

    try:
        res = requests.get(search_url, headers=headers, timeout=15)
        res.raise_for_status()
    except Exception as e:
        print(f"   ✖ Search failed: {e}")
        return None, None, None  # url, rating, rating_count

    soup = BeautifulSoup(res.text, "html.parser")

    # Find product anchors only
    anchors = soup.select("a[href*='/pd/']")
    candidates = []

    for a in anchors:
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href or not title:
            continue

        # get a surrounding "card" container (closest ancestor div/section/li that also contains this anchor)
        card = a
        for _ in range(4):  # climb up a few levels to find the card container
            card = card.parent
            if card is None:
                break
            # heuristic: card that contains title and other product details is usually div/section/li
            if card.name in ("div", "section", "li"):
                # found candidate card
                break

        # find rating inside the card (search for class containing "ReviewsAndRatings" or "ReviewsAndRating")
        rating = None
        rating_count = None
        rating_div = None
        if card:
            rating_div = card.find("div", class_=lambda c: c and "ReviewsAndRatings" in c)
            if not rating_div:
                # fallback: any span with numeric text near card
                rating_div = card.find("span", string=lambda s: s and re.match(r"^[0-5](\.\d)?$", s.strip()))
            if rating_div:
                # first numeric span inside rating_div is rating
                span = rating_div.find("span", string=lambda s: s and re.match(r"^[0-5](\.\d)?$", s.strip()))
                if span:
                    try:
                        rating = float(span.text.strip())
                    except:
                        rating = None
                # look for rating count e.g., "2693 Ratings" or "2.6K Ratings"
                rc_span = rating_div.find(string=lambda t: t and (("Ratings" in t) or re.search(r"\d", t)))
                if rc_span:
                    # extract number from text
                    m = re.search(r"([\d,\.kKmM]+)", rc_span)
                    if m:
                        rating_count = m.group(1).strip()
        # build absolute url
        full_url = href if href.startswith("http") else base_url + href

        # compute fuzzy scores against anchor title and the more granular product name
        score1 = fuzz.token_set_ratio((product + " " + brand).lower(), title.lower())
        score2 = fuzz.partial_ratio((product + " " + brand).lower(), title.lower())
        combined_score = max(score1, score2)

        candidates.append({
            "title": title,
            "url": full_url,
            "rating": rating,
            "rating_count": rating_count,
            "score": combined_score
        })

    if not candidates:
        print("   ✖ No product anchors found on search page.")
        return None, None, None

    # sort by presence of rating first, then by score
    candidates.sort(key=lambda c: ((0 if c["rating"] is None else 1), c["score"]), reverse=True)

    # pick top candidate, but ensure minimum match quality
    top = candidates[0]
    # require at least a reasonable threshold (e.g., 45-50). adjust if your data needs it.
    MIN_SCORE = 50
    if top["score"] < MIN_SCORE:
        # check if any candidate has rating and reasonable score
        for c in candidates:
            if c["rating"] is not None and c["score"] >= 40:
                top = c
                break
        else:
            # no good match found
            print(f"   ⚠️ Best candidate low score ({top['score']}%) — skipping")
            return None, None, None

    # debug print of top few candidates (optional)
    # print("   Top candidates:")
    # for c in candidates[:5]:
    #     print(f"     - {c['title'][:80]} | score={c['score']} | rating={c['rating']}")

    print(f"   ✓ Selected: {top['title'][:80]} | score={top['score']} | rating={top['rating']}")

    return top["url"], top["rating"], top["rating_count"]
# --- Main ---
df = pd.read_csv("BB_eggs_meat_fish.csv")

updated_ratings = []
for i, row in df.iterrows():
    product = row["product"]
    brand = row["brand"]
    print(f"[{i+1}/{len(df)}] Searching {product} ({brand}) ...")

    rating = get_rating_from_search(product, brand)
    if not rating:
        rating = row.get("rating_clean", 3)
    updated_ratings.append(rating)

    time.sleep(random.uniform(2, 5))  # delay to avoid blocking

df["rating_clean"] = updated_ratings
df.to_csv("BB_eggs_meat_fish_rated.csv", index=False)
print("\n✅ Done! Saved updated ratings to BB_eggs_meat_fish_rated.csv")

