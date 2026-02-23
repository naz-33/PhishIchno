"""URL feature extraction for phishing/safety classification."""
import re
from urllib.parse import urlparse
import ipaddress


def extract_url_features(url: str) -> dict:
    """
    Extract features from a URL for safety prediction.
    Returns a dict mapping feature names to values.
    Unknown features requested will default to 0.
    """
    if not url or not isinstance(url, str):
        url = ""

    # Normalize: ensure scheme for parsing
    url = url.strip()
    if url and "://" not in url:
        url = "http://" + url

    features = {}

    try:
        parsed = urlparse(url)
        domain = parsed.netloc or ""
        path = parsed.path or ""
        query = parsed.query or ""
    except Exception:
        parsed = None
        domain = ""
        path = ""
        query = ""

    # Structural
    features["url_length"] = len(url)
    features["domain_length"] = len(domain)
    features["path_length"] = len(path)
    features["query_length"] = len(query)

    # Aliases in case Colab uses different names
    features["url_len"] = features["url_length"]
    features["domain_len"] = features["domain_length"]
    features["path_len"] = features["path_length"]
    features["query_len"] = features["query_length"]

    # Counts
    features["num_dots"] = url.count(".")
    features["num_slashes"] = url.count("/")
    features["num_hyphens"] = url.count("-")
    features["num_underscores"] = url.count("_")
    features["num_ampersand"] = url.count("&")
    features["num_equals"] = url.count("=")
    features["num_question_marks"] = url.count("?")
    features["num_special_chars"] = sum(
        url.count(c) for c in "!@#$%^&*()_+-=[]{}|;':\",./<>?"
    )

    # Flags
    features["has_https"] = 1 if url.lower().startswith("https") else 0
    features["has_http"] = 1 if url.lower().startswith("http") else 0
    features["has_ip"] = 1 if _has_ip_address(domain or url) else 0
    features["has_at"] = 1 if "@" in url else 0
    features["has_port"] = 1 if ":" in (domain.split("@")[-1] if "@" in domain else domain) and domain.count(":") > 0 else 0

    # Subdomains
    features["num_subdomains"] = domain.count(".") - 1 if "." in domain and not _has_ip_address(domain) else 0
    features["num_subdomains"] = max(0, features["num_subdomains"])

    # Suspicious TLD
    suspicious_tlds = {
        ".tk", ".ml", ".ga", ".cf", ".gq",
        ".xyz", ".top", ".work", ".click", ".link",
        ".pw", ".cc", ".ws", ".buzz", ".rest"
    }
    tld = "." + domain.split(".")[-1].lower() if "." in domain else ""
    features["suspicious_tld"] = 1 if tld in suspicious_tlds else 0

    # Additional common features
    features["digit_count"] = sum(1 for c in url if c.isdigit())
    features["letter_count"] = sum(1 for c in url if c.isalpha())

    return features


def _has_ip_address(text: str) -> bool:
    """Check if text contains an IP address (v4 or v6)."""
    # Remove port if present
    if ":" in text and "]" not in text:
        text = text.split(":")[0]
    # IPv4 pattern
    ipv4_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    if re.search(ipv4_pattern, text):
        return True
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        pass
    return False


def build_feature_vector(feature_names: list, url: str) -> list:
    """
    Build a feature vector in the order expected by feature_names.
    Uses extract_url_features and fills missing features with 0.
    """
    extracted = extract_url_features(url)
    return [extracted.get(name, 0) for name in feature_names]
