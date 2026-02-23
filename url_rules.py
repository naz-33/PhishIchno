"""Rule-based URL safety checks: sensitive words, brand names, misspellings, and phishing heuristics."""
import re
from urllib.parse import urlparse
from difflib import SequenceMatcher

from url_features import extract_url_features

# High-risk credential words: increase phishing suspicion when combined with structural risks
HIGH_RISK_CREDENTIAL_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "reset", "password",
    "billing", "confirm", "authenticate", "account-update", "verify-account",
    "secure-login", "account-recovery", "credentials", "passwd", "verification",
]

# Neutral security-related words: alone should NOT cause UNSAFE
NEUTRAL_SECURITY_WORDS = [
    "security", "support", "help", "account", "service", "center",
]

# Well-known legitimate platforms; do NOT classify UNSAFE based only on keywords
TRUSTED_DOMAINS = {
    "github.com",
    "google.com",
    "microsoft.com",
    "microsoftonline.com",
    "paypal.com",
    "amazon.com",
    "facebook.com",
    "instagram.com",
    "apple.com",
    "linkedin.com",
    "netflix.com",
    "stripe.com",
    "aws.amazon.com",
    "dropbox.com",
    "twitter.com",
    "x.com",
}

# All sensitive/phishing-related words (for detection display)
SENSITIVE_WORDS = [
    "login", "log-in", "signin", "sign-in", "account", "verify", "verification",
    "password", "passwd", "credentials", "secure", "security", "update",
    "suspended", "locked", "confirm", "validation", "authenticate",
    "bank", "banking", "wire", "transfer", "payment", "paypal", "refund",
    "support", "helpdesk", "alert", "urgent", "immediate", "action-required",
    "click", "verify-account", "secure-login", "account-recovery",
    "admin", "administrator", "reset", "unlock", "restore", "billing",
]

# Commonly spoofed brands (phishers impersonate these)
BRAND_NAMES = [
    "google", "gmail", "yahoo", "microsoft", "outlook", "live", "office365",
    "amazon", "apple", "paypal", "netflix", "facebook", "instagram", "twitter",
    "linkedin", "dropbox", "adobe", "dhl", "fedex", "ups", "usps",
    "chase", "bankofamerica", "wellsfargo", "citibank", "capitalone",
    "ebay", "aliexpress", "walmart", "target", "bestbuy", "costco",
]

# Suspicious TLDs often used in phishing
SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".work", ".click",
    ".link", ".pw", ".cc", ".ws", ".buzz", ".rest", ".ru", ".info",
}


def _normalize_for_matching(text: str) -> str:
    """Lowercase and keep only alphanumeric for matching."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _generate_misspell_patterns(brand: str) -> list[re.Pattern]:
    """
    Generate regex patterns for common brand misspellings.
    Handles: 0->o, 1->i/l, 5->s, 4->a, 3->e, @->a
    """
    patterns = []
    # Character substitution map
    subs = {
        "o": "[o0]",
        "0": "[o0]",
        "i": "[i1l]",
        "l": "[i1l]",
        "1": "[i1l]",
        "s": "[s5]",
        "5": "[s5]",
        "a": "[a4@]",
        "4": "[a4]",
        "e": "[e3]",
        "3": "[e3]",
    }
    pattern_chars = []
    for c in brand.lower():
        pattern_chars.append(subs.get(c, re.escape(c)))
    pattern = "".join(pattern_chars)
    patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


def detect_sensitive_words(url: str) -> list[str]:
    """
    Detect sensitive/phishing-related words in the URL.
    Returns list of matched words.
    """
    if not url:
        return []
    url_lower = url.lower()
    found = []
    for word in SENSITIVE_WORDS:
        # Match whole word or as part of path segment (between / or . or start/end)
        pattern = r"(?:^|[/.\-_])" + re.escape(word) + r"(?:[/.\-_]|$|[?&#])"
        if re.search(pattern, url_lower):
            found.append(word)
    return found


def detect_high_risk_credential_words(url: str) -> list[str]:
    """Detect high-risk credential words that increase phishing suspicion."""
    if not url:
        return []
    url_lower = url.lower()
    found = []
    for word in HIGH_RISK_CREDENTIAL_WORDS:
        pattern = r"(?:^|[/.\-_])" + re.escape(word) + r"(?:[/.\-_]|$|[?&#])"
        if re.search(pattern, url_lower):
            found.append(word)
    return found


def detect_neutral_words(url: str) -> list[str]:
    """Detect neutral security-related words (alone should NOT cause UNSAFE)."""
    if not url:
        return []
    url_lower = url.lower()
    found = []
    for word in NEUTRAL_SECURITY_WORDS:
        pattern = r"(?:^|[/.\-_])" + re.escape(word) + r"(?:[/.\-_]|$|[?&#])"
        if re.search(pattern, url_lower):
            found.append(word)
    return found


def detect_brand_names(url: str) -> list[str]:
    """
    Detect known brand names in hostname or path.
    Returns list of matched brand names.
    """
    if not url or not isinstance(url, str):
        return []
    url = url.strip()
    if url and "://" not in url:
        url = "http://" + url
    try:
        parsed = urlparse(url)
        hostname = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        combined = hostname + "/" + path
    except Exception:
        combined = url.lower()
    found = []
    for brand in BRAND_NAMES:
        # Brand must appear as a distinct segment (surrounded by . / - or boundary)
        pattern = r"(?:^|[/.\-_])" + re.escape(brand) + r"(?:[/.\-_]|$|[?&#])"
        if re.search(pattern, combined):
            found.append(brand)
    return found


def detect_misspelled_brands(url: str) -> list[tuple[str, str]]:
    """
    Detect misspelled brand names (typosquatting).
    Uses character substitution patterns (0/o, 1/l, etc.) and similarity matching.
    Returns list of (matched_fragment, intended_brand).
    """
    if not url:
        return []
    url_lower = url.lower()
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # Extract hostname and path segments
    try:
        parsed = urlparse(url)
        hostname = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        segments = re.split(r"[/.\-_]", hostname + path)
    except Exception:
        segments = re.split(r"[/.\-_]", url_lower)

    for segment in segments:
        if len(segment) < 4:  # Skip very short segments
            continue
        for brand in BRAND_NAMES:
            if brand in segment:
                continue  # Exact match handled by detect_brand_names
            # Character-substitution pattern
            patterns = _generate_misspell_patterns(brand)
            for pat in patterns:
                m = pat.search(segment)
                if m:
                    match_str = m.group(0)
                    if len(match_str) >= len(brand) * 0.7:  # Require reasonable length
                        key = (match_str, brand)
                        if key not in seen:
                            seen.add(key)
                            found.append((match_str, brand))
                    break

    # Similarity-based for longer segments (catches typos like "gogle", "amazom")
    for segment in segments:
        if len(segment) < 4:
            continue
        for brand in BRAND_NAMES:
            if len(brand) < 4:
                continue
            # Compare segment to brand (or brand-sized sliding window)
            if len(segment) >= len(brand) * 0.8:
                ratio = SequenceMatcher(None, segment, brand).ratio()
                if 0.75 <= ratio < 0.99:  # Similar but not exact
                    key = (segment, brand)
                    if key not in seen:
                        seen.add(key)
                        found.append((segment, brand))
    return found


def _get_registered_domain(hostname: str) -> str:
    """Extract the registered domain (e.g. example.com from sub.example.com)."""
    parts = hostname.lower().split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname.lower()


def _is_trusted_domain(url: str) -> bool:
    """Check if registrable domain is a well-known legitimate platform."""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if url and "://" not in url:
        url = "http://" + url
    try:
        parsed = urlparse(url)
        hostname = (parsed.netloc or "").lower()
    except Exception:
        return False
    reg_domain = _get_registered_domain(hostname)
    return reg_domain in TRUSTED_DOMAINS


def detect_subdomain_tricks(url: str) -> list[str]:
    """
    Detect brand names in subdomains but NOT in the actual registered domain.
    e.g. google-security.com vs security.google.com (legit)
    Returns list of brands found only in subdomains.
    """
    if not url or not isinstance(url, str):
        return []
    url = url.strip()
    if url and "://" not in url:
        url = "http://" + url
    try:
        parsed = urlparse(url)
        hostname = (parsed.netloc or "").lower()
    except Exception:
        return []
    # Skip IP addresses
    feat = extract_url_features(url)
    if feat.get("has_ip"):
        return []
    reg_domain = _get_registered_domain(hostname)
    subdomain_part = hostname[: -len(reg_domain) - 1] if hostname.endswith("." + reg_domain) else ""
    found = []
    for brand in BRAND_NAMES:
        in_subdomain = brand in subdomain_part
        in_reg_domain = brand in reg_domain
        if in_subdomain and not in_reg_domain:
            found.append(brand)
    return found


def detect_suspicious_tld_plus_keywords(url: str) -> bool:
    """Suspicious TLD combined with high-risk credential keywords → UNSAFE."""
    if not url:
        return False
    feat = extract_url_features(url)
    if not feat.get("suspicious_tld"):
        return False
    url_lower = url.lower()
    for word in HIGH_RISK_CREDENTIAL_WORDS:
        if word in url_lower:
            return True
    return False


def detect_ip_address(url: str) -> bool:
    """Use of IP address instead of proper domain → UNSAFE."""
    feat = extract_url_features(url)
    return bool(feat.get("has_ip"))


def detect_excessive_hyphens_credentials(url: str) -> bool:
    """Excessive hyphens combined with high-risk credential words → UNSAFE."""
    if not url:
        return False
    feat = extract_url_features(url)
    num_hyphens = feat.get("num_hyphens", 0)
    if num_hyphens < 3:
        return False
    url_lower = url.lower()
    for word in HIGH_RISK_CREDENTIAL_WORDS:
        if word in url_lower:
            return True
    return False


def detect_punycode_encoded(url: str) -> bool:
    """Encoded, obfuscated, or punycode domains (xn--) → UNSAFE."""
    if not url:
        return False
    url_lower = url.lower()
    return "xn--" in url_lower


def detect_urgency_credential_words(url: str) -> list[str]:
    """High-risk credential words (login, verify, reset, etc.). Kept for backward compat."""
    return detect_high_risk_credential_words(url)


def is_unsafe_by_rules(url: str) -> tuple[bool, list[str]]:
    """
    Returns (is_unsafe, reasons).
    Structured reasoning: UNSAFE only when strong phishing indicators or
    suspicious combinations are present. Neutral words alone on trusted
    domains do NOT trigger UNSAFE.
    """
    reasons = []
    findings = run_rule_checks(url)

    # Structural indicators (always UNSAFE when present, except on trusted+no-others)
    has_typosquatting = bool(findings.get("misspelled_brands"))
    has_subdomain_trick = bool(findings.get("subdomain_tricks"))
    has_ip = bool(findings.get("has_ip"))
    has_punycode = bool(findings.get("punycode_encoded"))
    has_suspicious_tld_keywords = bool(findings.get("suspicious_tld_keywords"))
    has_excessive_hyphens_creds = bool(findings.get("excessive_hyphens_credentials"))

    structural_indicators = (
        has_typosquatting
        or has_subdomain_trick
        or has_ip
        or has_punycode
        or has_suspicious_tld_keywords
        or has_excessive_hyphens_creds
    )

    trusted = _is_trusted_domain(url)

    if trusted:
        # On trusted domains: only UNSAFE if additional phishing indicators present
        # Keywords alone (high-risk or neutral) do NOT trigger UNSAFE
        if structural_indicators:
            if has_typosquatting:
                reasons.append("Typosquatting or misspelled brand names")
            if has_subdomain_trick:
                reasons.append("Subdomain tricks (brand in subdomain, not actual domain)")
            if has_suspicious_tld_keywords:
                reasons.append("Suspicious TLD combined with credential keywords")
            if has_ip:
                reasons.append("IP address used instead of proper domain")
            if has_excessive_hyphens_creds:
                reasons.append("Excessive hyphens with credential-related words")
            if has_punycode:
                reasons.append("Encoded, obfuscated, or punycode domain (xn--)")
            return True, reasons
        return False, []

    # Untrusted domain: strong indicators or suspicious combos → UNSAFE
    if has_typosquatting:
        reasons.append("Typosquatting or misspelled brand names")
    if has_subdomain_trick:
        reasons.append("Subdomain tricks (brand in subdomain, not actual domain)")
    if has_ip:
        reasons.append("IP address used instead of proper domain")
    if has_punycode:
        reasons.append("Encoded, obfuscated, or punycode domain (xn--)")
    if has_suspicious_tld_keywords:
        reasons.append("Suspicious TLD combined with credential keywords")
    if has_excessive_hyphens_creds:
        reasons.append("Excessive hyphens with credential-related words")

    if reasons:
        return True, reasons
    return False, []


def run_rule_checks(url: str) -> dict:
    """
    Run all rule-based checks and return findings.
    Returns dict with keys for display and is_unsafe_by_rules evaluation.
    """
    subdomain_tricks = detect_subdomain_tricks(url)
    high_risk = detect_high_risk_credential_words(url)
    return {
        "sensitive_words": detect_sensitive_words(url),
        "brand_names": detect_brand_names(url),
        "misspelled_brands": detect_misspelled_brands(url),
        "subdomain_tricks": subdomain_tricks,
        "suspicious_tld_keywords": detect_suspicious_tld_plus_keywords(url),
        "has_ip": detect_ip_address(url),
        "excessive_hyphens_credentials": detect_excessive_hyphens_credentials(url),
        "punycode_encoded": detect_punycode_encoded(url),
        "urgency_credential_words": high_risk,
        "high_risk_credential_words": high_risk,
        "neutral_words": detect_neutral_words(url),
    }
