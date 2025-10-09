
import requests, time, re, json
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tqdm import tqdm
from urllib.parse import quote_plus
import google.generativeai as genai

# ⚙️ configure Gemini once here
genai.configure(api_key="AIzaSyCVlbh7euN0VtZIepD-NtI3PTcZtoQ95P4")
model = genai.GenerativeModel("models/gemini-2.5-flash")

# include your scraper, helpers, etc.
# (copy from your notebook: get_drug_page_by_name, extract_fields_from_html, etc.)

UA = UserAgent()
BASE = "https://www.drugs.com"
HEADERS = lambda: {"User-Agent": UA.random}

def get_or_generate_medicine_report(drug_name):
    scraped = get_drug_page_by_name(drug_name, save_json=False)

    def is_meaningful(data):
        if not data:
            return False
        # if most fields are 'Not found' → treat as empty
        not_found_count = sum(1 for v in data.values() if str(v).lower() == "not found")
        return not_found_count < len(data) * 0.7  # at least 30% real info

    # ✅ CASE 1: Found and meaningful → summarize with Gemini
    if scraped["found"] and is_meaningful(scraped.get("data")):
        data = scraped["data"]

        # Join all text fields into one raw body
        raw_text = "\n".join(
            f"{k}: {v}" for k, v in data.items() if v and v.lower() != "not found"
        )

        # Optional: truncate extremely long content
        if len(raw_text) > 6000:
            raw_text = raw_text[:6000] + "... (truncated)"

        # Ask Gemini to summarize the scraped content
        prompt = f"""
        You are a multilingual medical assistant.

        The following text contains partially scraped drug data from Drugs.com for "{drug_name}".
        Please complete and clean it into a structured, bilingual (English + Arabic) medicine report.

        Always include these fields:
        1. Drug Name
        2. Active Ingredient
        3. Indications (uses)
        4. Dosage
        5. Contraindications
        6. Side Effects
        7. Interactions
        8. Alternatives
        9. Final Warning: "Not medical advice — consult a doctor before use."

        - Fill in any missing sections intelligently using your medical knowledge.
        - Output a well-formatted report (not JSON) with clear sections:
          🩺 **Medicine Report — تقرير الدواء**
          🇬🇧 English Section
          🇸🇦 Arabic Section

        Here is the raw extracted content:
        {raw_text}

        If needed, also infer details from general medical knowledge about {drug_name}.
        """

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "top_p": 0.9},
        )
        return response.text

    # 🚨 CASE 2: Not found or empty → generate from scratch
    else:
        prompt = f"""
        You are a multilingual medical assistant.
        Generate a **bilingual medicine report** (English + Arabic) for the drug "{drug_name}".
        The report must include the following sections in **both English and Arabic**,
        keeping the same order and structure:

        1. Drug Name
        2. Active Ingredient
        3. Indications (uses)
        4. Dosage (for adults and pediatrics if available)
        5. Contraindications
        6. Side Effects
        7. Interactions (if important)
        8. Alternatives (with examples)
        9. A final WARNING: "Not medical advice — consult a doctor before use."

        Format clearly using headings and bullet points.
        Example:
        ---
        **English Section**
        ...
        **Arabic Section**
        ...
        """

        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "top_p": 0.9},
        )
        return response.text


# ---------- main flow ----------
def get_drug_page_by_name(name, save_json=True, save_sqlite=False, delay=1.2):
    slug = slugify(name)
    candidates = [
        f"{BASE}/drugs/{slug}.html",
        f"{BASE}/mtm/{slug}.html",
        f"{BASE}/{slug}.html",
    ]

    html = None
    final_url = None

    # try candidate URLs
    for url in candidates:
        r = try_fetch(url)
        if r and r.status_code == 200 and 'html' in r.headers.get('Content-Type',''):
            html = r.text; final_url = url; break
        time.sleep(delay)

    # if not found, try search results
    if not html:
        links = search_drugsdotcom(name)
        for url in links:
            r = try_fetch(url)
            if r and r.status_code == 200 and 'html' in r.headers.get('Content-Type',''):
                html = r.text; final_url = url; break
            time.sleep(delay)

    result = {"query": name, "found": False, "url": final_url, "data": None}

    if not html:
        # not found on site
        return result

    # extract fields
    fields = extract_fields_from_html(html)
    result["found"] = True
    result["data"] = fields
    result["url"] = final_url

    # save to JSON (append)
    if save_json:
        try:
            fname = "drugs_output.json"
            try:
                existing = json.load(open(fname, "r", encoding="utf-8"))
            except Exception:
                existing = []
            existing.append(result)
            json.dump(existing, open(fname, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as e:
            print("Warning: failed saving JSON:", e)

    return result

def extract_fields_from_html(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # fields and heuristics - adjust keywords/patterns if you find site variants
    fields = {}
    fields['drug_name'] = clean_text(soup.find('h1').get_text(" ", strip=True)) if soup.find('h1') else "Not found"
    fields['What is'] = (get_by_headings(soup, ['what is', 'overview', 'description'])
                         or fallback_regex_search(full_text, [r'what is (?:[^\?\.]+)\?(.+?)(?:\n|$)', r'Overview:?\s*(.+?)(?:\n|$)'])
                         or "Not found")
    fields['What is used for'] = (get_by_headings(soup, ['uses', 'used for', 'indication', 'indications'])
                                 or fallback_regex_search(full_text, [r'Used for:?(.+?)(?:\n|$)', r'Indication[s]?:\s*(.+?)(?:\n|$)'])
                                 or "Not found")
    fields['Important information'] = (get_by_headings(soup, ['important information', 'important', 'warnings', 'warnings and precautions'])
                                       or fallback_regex_search(full_text, [r'Important information:?\s*(.+?)(?:\n|$)', r'Warnings:?\s*(.+?)(?:\n|$)'])
                                       or "Not found")
    fields['Who should not take'] = (get_by_headings(soup, ['who should not take', 'contraindications', 'do not take'])
                                     or fallback_regex_search(full_text, [r'Who should not take(?:.+?):\s*(.+?)(?:\n|$)', r'Contraindication[s]?:\s*(.+?)(?:\n|$)'])
                                     or "Not found")
    fields['What should I tell my doctor'] = (get_by_headings(soup, ['before taking', 'tell your doctor', 'precautions'])
                                              or fallback_regex_search(full_text, [r'Before taking(?:.+?):\s*(.+?)(?:\n|$)', r'Tell your doctor(?:.+?):\s*(.+?)(?:\n|$)'])
                                              or "Not found")
    fields['How should I take'] = (get_by_headings(soup, ['how to take', 'dosage', 'administration', 'how should i take'])
                                   or fallback_regex_search(full_text, [r'Dosage and administration:?\s*(.+?)(?:\n|$)', r'Dosage:?\s*(.+?)(?:\n|$)'])
                                   or "Not found")
    fields['What are the side effects'] = (get_by_headings(soup, ['side effect', 'adverse'])
                                           or fallback_regex_search(full_text, [r'Side effects:?\s*(.+?)(?:\n|$)'])
                                           or "Not found")
    fields['Interactions'] = (get_by_headings(soup, ['interaction', 'interactions'])
                              or fallback_regex_search(full_text, [r'Interaction[s]?:\s*(.+?)(?:\n|$)'])
                              or "Not found")
    fields['Storage'] = (get_by_headings(soup, ['storage', 'store'])
                         or fallback_regex_search(full_text, [r'Storage:?\s*(.+?)(?:\n|$)'])
                         or "Not found")
    fields['What are the ingredients'] = (get_by_headings(soup, ['ingredient', 'composition', 'active ingredient'])
                                          or fallback_regex_search(full_text, [r'Active ingredient[s]?:\s*(.+?)(?:\n|$)', r'Ingredients:?\s*(.+?)(?:\n|$)'])
                                          or "Not found")
    fields['alt'] = (get_by_headings(soup, ['brand names', 'also known as', 'alternative', 'other names'])
                     or fallback_regex_search(full_text, [r'Brand names:?\s*(.+?)(?:\n|$)', r'Also known as:?\s*(.+?)(?:\n|$)'])
                     or "Not found")
    return fields

def slugify(name):
    s = name.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s

def try_fetch(url, timeout=12):
    """Fetch a page with realistic headers and handle 'short but valid' pages."""
    try:
        r = requests.get(url, headers=HEADERS(), timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or "html" not in r.headers.get("Content-Type", ""):
            return None

        text = r.text.strip()
        # Only retry if truly empty or a JS warning (not just a short page like Plan B)
        if len(text) < 1000 or "Please enable JavaScript" in text:
            if "/mtm/" in url:
                return try_fetch(url.replace("/mtm/", "/"), timeout)
            elif "/generic/" in url:
                return try_fetch(url.replace("/generic/", "/"), timeout)
        return r
    except Exception:
        return None

def search_drugsdotcom(query):
    """Use site's search page to find candidate links (fallback if slug fails)"""
    url = f"{BASE}/search.php?searchterm={quote_plus(query)}"
    r = try_fetch(url)
    if not r or r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    # common result link selectors (flexible)
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if href.startswith("/mtm/") or href.startswith("/drugs/") or href.startswith("/drug/") or href.endswith(".html"):
            full = BASE + href if href.startswith("/") else href
            out.append(full)
    # dedupe while preserving order
    seen = set(); res = []
    for u in out:
        if u not in seen:
            seen.add(u); res.append(u)
    return res

# ---------- robust extractor (search headings + fallback search) ----------
def clean_text(s):
    return re.sub(r'\s+', ' ', s.strip()) if s else ""

def get_by_headings(soup, keywords):
    for tag in soup.find_all(['h1','h2','h3','h4']):
        text = tag.get_text(" ", strip=True).lower()
        if any(k in text for k in keywords):
            parts = []
            for sib in tag.find_next_siblings():
                if sib.name and sib.name.startswith('h'):
                    break
                if sib.name in ['p','ul','div','table','ol']:
                    parts.append(sib.get_text(" ", strip=True))
            joined = " ".join(parts).strip()
            if joined:
                return clean_text(joined)
    return None

def fallback_regex_search(full_text, patterns):
    for p in patterns:
        m = re.search(p, full_text, flags=re.IGNORECASE|re.DOTALL)
        if m:
            return clean_text(m.group(1))
    return None