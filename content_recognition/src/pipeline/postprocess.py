from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def extract_email(text: str) -> str:
    m = EMAIL_RE.search(text)
    return m.group(0) if m else ""

def extract_phone(text: str) -> str:
    # 粗提取：把常见符号保留，最后再压缩空格
    m = PHONE_RE.search(text)
    if not m:
        return ""
    t = m.group(0)
    t = normalize_spaces(t)
    return t

def merge_lines(lines: List[str]) -> str:
    # 地址/公司可能多行：用空格连接；你也可以改成 '\n'
    return normalize_spaces(" ".join([x for x in lines if x.strip()]))

def postprocess_fields(raw: Dict[str, List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}

    out["name"] = merge_lines(raw.get("name", []))
    out["company"] = merge_lines(raw.get("company", []))
    out["address"] = merge_lines(raw.get("address", []))

    email_text = merge_lines(raw.get("email", []))
    phone_text = merge_lines(raw.get("phone", []))

    out["email"] = extract_email(email_text) if email_text else ""
    out["phone"] = extract_phone(phone_text) if phone_text else ""

    return out
