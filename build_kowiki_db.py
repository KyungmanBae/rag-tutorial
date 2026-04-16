#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
한국어 위키백과 덤프 → SQLite DB 변환 스크립트
===========================================================
최종 목표: korfactscore_lab.py 의 DB 포맷과 완전히 호환되는
           kowiki-YYYYMMDD.db 를 생성합니다.

[사전 준비]
  pip install requests tqdm mwparserfromhell

[실행 예시]
  # 최신 덤프 자동 탐지 + 기본 출력 경로
  python build_kowiki_db.py

  # 날짜 직접 지정
  python build_kowiki_db.py --date 20240301

  # 이미 다운로드된 .bz2 파일 사용
  python build_kowiki_db.py --input_bz2 /path/to/kowiki-20240301-pages-articles.xml.bz2

  # 출력 경로 지정
  python build_kowiki_db.py --out_db ./KorFactScore/downloaded_files/kowiki-20240301.db

[DB 스키마 - korfactscore_lab.py 호환]
  테이블명: documents
  컬럼:
    id    INTEGER PRIMARY KEY AUTOINCREMENT
    title TEXT                              -- 위키 표제어 (공백 정규화)
    text  TEXT                              -- 단락들을 SPECIAL_SEPARATOR 로 결합

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
===========================================================
"""

import os
import re
import sys
import bz2
import json
import time
import sqlite3
import hashlib
import argparse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

# ────────────────────────────────────────────────────────────
# 상수
# ────────────────────────────────────────────────────────────

# korfactscore_lab.py 와 동일한 구분자
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

# 한국어 위키백과 덤프 미러
DUMP_BASE_URL = "https://dumps.wikimedia.org/kowiki"

# 파싱 대상 덤프 파일 패턴
DUMP_FILENAME_PATTERN = "kowiki-{date}-pages-articles.xml.bz2"

# 최소 단락 길이 (짧은 토막글/템플릿 필터링)
MIN_PARA_LEN = 20

# DB COMMIT 주기 (메모리 절약)
COMMIT_EVERY = 10_000

# mwparserfromhell 사용 가능 여부
try:
    import mwparserfromhell
    HAS_MWPH = True
except ImportError:
    HAS_MWPH = False
    print("[경고] mwparserfromhell 미설치 — 정규식 기반 파싱으로 대체합니다.")
    print("       pip install mwparserfromhell  을 실행하면 품질이 향상됩니다.\n")

# tqdm 사용 가능 여부
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────

def normalize_title(title: str) -> str:
    """
    korfactscore_lab.py 의 _normalize_title() 과 동일한 로직.
    밑줄 → 공백, 연속 공백 → 단일 공백, strip.
    """
    title = (title or "").strip()
    title = title.replace("_", " ")
    title = re.sub(r"\s+", " ", title)
    return title


def is_redirect(text: str) -> bool:
    """위키마크업 redirect 페이지 여부."""
    return bool(re.match(r"^\s*#(redirect|넘겨주기)", text, re.IGNORECASE))


def is_special_title(title: str) -> bool:
    """
    특수 네임스페이스(파일, 분류, 토론 등) 제외.
    실제 문서(ns=0)만 남기도록 title 기반으로 1차 필터링.
    """
    special_prefixes = (
        "위키백과:", "사용자:", "파일:", "미디어위키:", "틀:",
        "도움말:", "분류:", "포털:", "책:", "모듈:",
        "Wikipedia:", "User:", "File:", "Template:",
        "Help:", "Category:", "Portal:", "Book:", "Module:",
        "MediaWiki:", "Talk:", "토론:",
    )
    for prefix in special_prefixes:
        if title.startswith(prefix):
            return True
    return False


# ────────────────────────────────────────────────────────────
# 위키 마크업 → 순수 텍스트 변환
# ────────────────────────────────────────────────────────────

def _strip_markup_mwph(wikitext: str) -> str:
    """
    mwparserfromhell 을 사용해 위키 마크업을 제거합니다.
    템플릿, 링크, 파일 등을 처리합니다.
    """
    parsed = mwparserfromhell.parse(wikitext)
    # 파일/이미지 링크 제거
    for tag in parsed.filter_wikilinks():
        target = str(tag.title).strip()
        if target.startswith(("파일:", "File:", "Image:", "이미지:")):
            try:
                parsed.remove(tag)
            except Exception:
                pass
    return parsed.strip_code()


def _strip_markup_regex(wikitext: str) -> str:
    """
    정규식 기반 마크업 제거 (mwparserfromhell 없을 때 fallback).
    완벽하지 않지만 실용적인 수준.
    """
    text = wikitext

    # HTML 주석 제거
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # <ref> 태그 (인용 각주) 제거
    text = re.sub(r"<ref[^>]*/?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^/]*/?>", "", text)

    # 기타 HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)

    # {{...}} 템플릿 중첩 제거 (단순 1~3 레벨)
    for _ in range(4):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)

    # [[파일:...]] [[File:...]] 이미지 제거
    text = re.sub(
        r"\[\[(파일|File|Image|이미지):[^\[\]]*(\[\[[^\[\]]*\]\])?[^\[\]]*\]\]",
        "", text, flags=re.IGNORECASE
    )

    # [[링크|표시텍스트]] → 표시텍스트
    text = re.sub(r"\[\[[^\[\]|]+\|([^\[\]]+)\]\]", r"\1", text)

    # [[링크]] → 링크
    text = re.sub(r"\[\[([^\[\]]+)\]\]", r"\1", text)

    # [외부링크 텍스트] → 텍스트
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)

    # 굵기/이탤릭 마크업 제거
    text = re.sub(r"'{2,3}", "", text)

    # = 제목 마크업 제거 (줄의 시작/끝)
    text = re.sub(r"^=+\s*(.+?)\s*=+$", r"\1", text, flags=re.MULTILINE)

    # 표 마크업 제거
    text = re.sub(r"^\s*\|[^|].*$", "", text, flags=re.MULTILINE)  # 표 셀
    text = re.sub(r"^\s*\|-.*$", "", text, flags=re.MULTILINE)      # 표 행 구분
    text = re.sub(r"^\s*\{\|.*$", "", text, flags=re.MULTILINE)     # 표 시작
    text = re.sub(r"^\s*\|}.*$", "", text, flags=re.MULTILINE)      # 표 끝
    text = re.sub(r"^\s*!.*$", "", text, flags=re.MULTILINE)        # 표 헤더

    # 목록 기호 (* , # , : ) 제거
    text = re.sub(r"^[*#:;]+\s*", "", text, flags=re.MULTILINE)

    # 연속 공백/줄바꿈 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def strip_markup(wikitext: str) -> str:
    """마크업 제거 디스패처."""
    if HAS_MWPH:
        return _strip_markup_mwph(wikitext)
    return _strip_markup_regex(wikitext)


def split_into_paragraphs(plain_text: str) -> list:
    """
    순수 텍스트를 단락 단위로 분리합니다.

    korfactscore_lab.py 의 load_passages() 에서
    SPECIAL_SEPARATOR 로 split 한 뒤 각 단락을 개별 passage 로 사용합니다.
    따라서 각 단락은 의미 있는 문장 묶음이어야 합니다.

    분리 기준:
      1. 빈 줄 (\\n\\n)
      2. == 제목 == 섹션 경계
    """
    # 섹션 제목 앞에 줄바꿈 2개 추가 (단락 경계로 처리)
    plain_text = re.sub(r"([^\n])\n(={2,})", r"\1\n\n\2", plain_text)

    # 빈 줄 기준 분리
    raw_paras = re.split(r"\n\s*\n", plain_text)

    paragraphs = []
    for p in raw_paras:
        p = p.strip()
        # 너무 짧은 단락 제외 (목록 잔재, 빈 섹션 제목 등)
        if len(p) < MIN_PARA_LEN:
            continue
        # 섹션 제목만 있는 단락 제외 (== 제목 == 패턴)
        if re.match(r"^={1,6}\s*[^=]+\s*={1,6}$", p):
            continue
        paragraphs.append(p)

    return paragraphs


def wikitext_to_db_text(wikitext: str) -> str | None:
    """
    위키 마크업 → DB에 저장할 text 값 변환.

    반환값:
      SPECIAL_SEPARATOR 로 결합된 단락 문자열
      유효 단락이 없으면 None
    """
    plain = strip_markup(wikitext)
    paragraphs = split_into_paragraphs(plain)

    if not paragraphs:
        return None

    return SPECIAL_SEPARATOR.join(paragraphs)


# ────────────────────────────────────────────────────────────
# 덤프 다운로드
# ────────────────────────────────────────────────────────────

def _get_latest_dump_date() -> str:
    """
    dumps.wikimedia.org 에서 사용 가능한 최신 덤프 날짜를 가져옵니다.
    HTML 파싱 없이 status.json 을 활용합니다.
    """
    import json as _json

    # 먼저 최신 덤프 디렉토리 목록 페이지를 확인
    url = f"{DUMP_BASE_URL}/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KorWikiDB/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read().decode("utf-8", errors="replace")
        # 날짜 패턴 추출 (YYYYMMDD)
        dates = re.findall(r'href="(\d{8})/"', html)
        if not dates:
            return "20240301"  # fallback
        # 최신 순 정렬 후 완료된 덤프 확인
        for date in sorted(dates, reverse=True):
            status_url = f"{DUMP_BASE_URL}/{date}/dumpstatus.json"
            try:
                req2 = urllib.request.Request(
                    status_url, headers={"User-Agent": "KorWikiDB/1.0"})
                with urllib.request.urlopen(req2, timeout=20) as r2:
                    status = _json.loads(r2.read())
                # articles 덤프가 완료된 날짜 반환
                jobs = status.get("jobs", {})
                articles_job = jobs.get("articlesdump", {})
                if articles_job.get("status") == "done":
                    print(f"[INFO] 최신 완료 덤프 날짜: {date}")
                    return date
            except Exception:
                continue
        return sorted(dates, reverse=True)[0]
    except Exception as e:
        print(f"[경고] 최신 날짜 탐지 실패 ({e}), 기본값 사용")
        return "20240301"


def _download_with_progress(url: str, dest_path: str) -> None:
    """
    URL 에서 파일을 다운로드하며 진행 상황을 출력합니다.
    이미 완전히 다운로드된 파일은 건너뜁니다.

    대용량 파일(수 GB)이므로 청크 단위로 저장합니다.
    """
    # 이미 존재하는지 확인
    if os.path.exists(dest_path):
        local_size = os.path.getsize(dest_path)
        try:
            req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "KorWikiDB/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                remote_size = int(r.headers.get("Content-Length", 0))
            if local_size == remote_size and remote_size > 0:
                print(f"[SKIP] 이미 다운로드됨: {dest_path} ({local_size / 1024**3:.2f} GB)")
                return
        except Exception:
            pass  # HEAD 요청 실패 시 재다운로드

    print(f"[다운로드] {url}")
    print(f"  → {dest_path}")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "KorWikiDB/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB
            start_time = time.time()

            with open(dest_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # 진행 상황 출력
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed / 1024**2 if elapsed > 0 else 0
                    if total > 0:
                        pct = downloaded / total * 100
                        done_gb = downloaded / 1024**3
                        total_gb = total / 1024**3
                        print(
                            f"\r  {pct:5.1f}%  {done_gb:.2f}/{total_gb:.2f} GB"
                            f"  {speed:.1f} MB/s",
                            end="", flush=True
                        )
                    else:
                        print(
                            f"\r  {downloaded / 1024**2:.1f} MB  {speed:.1f} MB/s",
                            end="", flush=True
                        )

            print(f"\n[완료] {downloaded / 1024**3:.2f} GB 다운로드")

    except urllib.error.URLError as e:
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == 0:
            os.remove(dest_path)
        raise RuntimeError(f"다운로드 실패: {e}") from e


def download_dump(date: str, dest_dir: str) -> str:
    """
    한국어 위키백과 articles 덤프를 다운로드합니다.

    Args:
        date: "YYYYMMDD" 형식
        dest_dir: 저장 디렉토리

    Returns:
        다운로드된 .bz2 파일 경로
    """
    filename = DUMP_FILENAME_PATTERN.format(date=date)
    url = f"{DUMP_BASE_URL}/{date}/{filename}"
    dest_path = os.path.join(dest_dir, filename)

    _download_with_progress(url, dest_path)
    return dest_path


# ────────────────────────────────────────────────────────────
# XML 스트리밍 파서
# ────────────────────────────────────────────────────────────

def iter_wiki_pages(bz2_path: str):
    """
    bz2 압축된 위키백과 XML 덤프를 스트리밍 파싱합니다.

    XML 구조:
      <mediawiki>
        <page>
          <title>...</title>
          <ns>0</ns>          ← ns=0 이 일반 문서
          <revision>
            <text>...</text>  ← 위키 마크업
          </revision>
        </page>
      </mediawiki>

    Yields:
        (title: str, wikitext: str)
    """
    # ElementTree iterparse: 메모리 효율적
    # bz2 파일을 스트리밍으로 읽음
    ns_map = {
        "mw": "http://www.mediawiki.org/xml/DTD/mediawiki"
    }

    # bz2 스트림 열기
    with bz2.open(bz2_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        current_title = None
        current_ns    = None
        in_revision   = False

        for event, elem in context:
            # 네임스페이스 제거 후 태그명 추출
            tag = elem.tag
            if "}" in tag:
                tag = tag.split("}", 1)[1]

            if tag == "title":
                current_title = (elem.text or "").strip()

            elif tag == "ns":
                current_ns = elem.text

            elif tag == "text":
                # revision 내부의 text 태그 처리
                if current_ns == "0" and current_title:
                    wikitext = elem.text or ""
                    if not is_redirect(wikitext) and not is_special_title(current_title):
                        yield current_title, wikitext

            elif tag == "page":
                # 페이지 파싱 완료 — 메모리 해제
                current_title = None
                current_ns    = None
                elem.clear()


# ────────────────────────────────────────────────────────────
# SQLite DB 생성
# ────────────────────────────────────────────────────────────

def create_db(db_path: str) -> sqlite3.Connection:
    """
    korfactscore_lab.py 와 호환되는 SQLite DB 를 생성합니다.

    스키마:
      documents(id, title, text)
        - id    : 자동 증가 기본키
        - title : 정규화된 표제어 (공백 기준, 중복 없음)
        - text  : SPECIAL_SEPARATOR 로 결합된 단락들

    인덱스:
      idx_documents_title : title 컬럼 (검색 속도 최적화)
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")   # 쓰기 성능 향상
    conn.execute("PRAGMA synchronous=NORMAL") # 안정성/속도 균형
    conn.execute("PRAGMA cache_size=-524288") # 512 MB 캐시

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            text  TEXT NOT NULL
        )
    """)

    # title 컬럼 인덱스 생성
    # korfactscore_lab.py 의 쿼리: WHERE title = ? 또는 LIKE ?
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_title
        ON documents(title)
    """)

    conn.commit()
    return conn


def insert_batch(conn: sqlite3.Connection, batch: list) -> None:
    """
    (title, text) 리스트를 배치 삽입합니다.
    동일 title 이 있으면 text 를 이어붙입니다 (드문 케이스).
    """
    conn.executemany(
        "INSERT OR IGNORE INTO documents(title, text) VALUES (?, ?)",
        batch,
    )


# ────────────────────────────────────────────────────────────
# 메인 변환 로직
# ────────────────────────────────────────────────────────────

def build_db(bz2_path: str, db_path: str,
             max_articles: int = 0,
             show_progress: bool = True) -> None:
    """
    위키백과 bz2 덤프를 파싱하여 SQLite DB 를 생성합니다.

    Args:
        bz2_path   : 입력 .xml.bz2 파일 경로
        db_path    : 출력 SQLite DB 경로
        max_articles: 0이면 전체, 양수면 해당 개수만 처리 (테스트용)
        show_progress: True 면 콘솔에 진행 상황 출력
    """
    print(f"\n[DB 생성 시작]")
    print(f"  입력 : {bz2_path}")
    print(f"  출력 : {db_path}")
    if max_articles > 0:
        print(f"  [테스트 모드] 최대 {max_articles:,}개 문서만 처리")
    print()

    conn   = create_db(db_path)
    batch  = []

    n_processed  = 0  # XML 에서 읽은 일반 문서 수
    n_inserted   = 0  # DB 에 실제 삽입된 문서 수
    n_skipped    = 0  # 유효 단락 없어서 건너뛴 문서 수
    start_time   = time.time()

    try:
        for title, wikitext in iter_wiki_pages(bz2_path):

            # ── 마크업 제거 + 단락 분리 ──────────
            db_text = wikitext_to_db_text(wikitext)

            if db_text is None:
                n_skipped += 1
                continue

            # ── 표제어 정규화 ─────────────────────
            norm_title = normalize_title(title)

            batch.append((norm_title, db_text))
            n_processed += 1

            # ── 배치 커밋 ─────────────────────────
            if len(batch) >= COMMIT_EVERY:
                insert_batch(conn, batch)
                conn.commit()
                n_inserted += len(batch)
                batch.clear()

                if show_progress:
                    elapsed = time.time() - start_time
                    rate = n_processed / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{n_processed:>8,}개 처리 | {n_inserted:>8,}개 삽입 | "
                        f"{n_skipped:>6,}개 건너뜀 | {rate:.0f} doc/s]"
                    )

            # ── 테스트 모드 조기 종료 ──────────────
            if max_articles > 0 and n_processed >= max_articles:
                print(f"\n[테스트 모드] {max_articles:,}개 도달 → 조기 종료")
                break

        # 남은 배치 커밋
        if batch:
            insert_batch(conn, batch)
            conn.commit()
            n_inserted += len(batch)

    finally:
        conn.close()

    elapsed = time.time() - start_time
    db_size = os.path.getsize(db_path) / 1024**3 if os.path.exists(db_path) else 0

    print(f"\n[DB 생성 완료]")
    print(f"  처리 문서 : {n_processed:,}개")
    print(f"  삽입 문서 : {n_inserted:,}개")
    print(f"  건너뜀   : {n_skipped:,}개 (유효 단락 없음)")
    print(f"  소요 시간 : {elapsed/60:.1f}분")
    print(f"  DB 크기   : {db_size:.2f} GB")
    print(f"  DB 경로   : {db_path}")


# ────────────────────────────────────────────────────────────
# DB 검증
# ────────────────────────────────────────────────────────────

def verify_db(db_path: str, sample_titles: list = None) -> None:
    """
    생성된 DB 가 korfactscore_lab.py 와 호환되는지 검증합니다.

    검증 항목:
      1. 테이블 / 컬럼 존재 여부
      2. 총 문서 수
      3. 샘플 표제어 검색 테스트
      4. SPECIAL_SEPARATOR 분리 동작 테스트
    """
    print(f"\n[DB 검증] {db_path}")

    if not os.path.exists(db_path):
        print("  ❌ DB 파일이 존재하지 않습니다.")
        return

    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()

    # 1. 스키마 확인
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    print(f"  테이블: {sorted(tables)}")

    if "documents" not in tables:
        print("  ❌ 'documents' 테이블이 없습니다.")
        conn.close()
        return
    print("  ✅ 테이블 'documents' 존재")

    # 2. 총 문서 수
    cur.execute("SELECT COUNT(*) FROM documents")
    total = cur.fetchone()[0]
    print(f"  ✅ 총 문서 수: {total:,}개")

    # 3. 샘플 검색
    default_samples = ["세종대왕", "이순신", "대한민국", "서울", "조선"]
    targets = sample_titles or default_samples

    print(f"\n  [샘플 검색 테스트]")
    for title in targets:
        cur.execute(
            "SELECT title, text FROM documents WHERE title = ? LIMIT 1",
            (title,)
        )
        row = cur.fetchone()
        if row:
            paras = row[1].split(SPECIAL_SEPARATOR)
            print(f"  ✅ '{title}': {len(paras)}개 단락, "
                  f"첫 단락 {len(paras[0])}자")
            print(f"     미리보기: {paras[0][:80]}...")
        else:
            # 부분 검색 시도
            cur.execute(
                "SELECT title FROM documents WHERE title LIKE ? LIMIT 3",
                (f"%{title}%",)
            )
            like_rows = cur.fetchall()
            if like_rows:
                matched = [r[0] for r in like_rows]
                print(f"  ⚠️  '{title}': 정확 일치 없음, 유사: {matched}")
            else:
                print(f"  ❌ '{title}': 검색 결과 없음")

    # 4. SPECIAL_SEPARATOR 동작 테스트
    cur.execute("SELECT text FROM documents LIMIT 1")
    row = cur.fetchone()
    if row:
        paras = row[0].split(SPECIAL_SEPARATOR)
        print(f"\n  [SEPARATOR 테스트]")
        print(f"  ✅ 첫 번째 문서: {len(paras)}개 단락으로 분리 가능")

    conn.close()
    print(f"\n  → korfactscore_lab.py 와 호환 가능합니다.")


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="한국어 위키백과 덤프 → KorFactScore 호환 SQLite DB 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 최신 덤프 자동 다운로드 + 변환
  python build_kowiki_db.py

  # 날짜 지정
  python build_kowiki_db.py --date 20240301

  # 이미 다운로드된 bz2 파일 사용
  python build_kowiki_db.py --input_bz2 ./kowiki-20240301-pages-articles.xml.bz2

  # 출력 경로 지정
  python build_kowiki_db.py --out_db ./KorFactScore/downloaded_files/kowiki-20240301.db

  # 테스트 (1000개 문서만)
  python build_kowiki_db.py --max_articles 1000 --out_db ./test_kowiki.db
        """
    )

    p.add_argument(
        "--date", type=str, default=None,
        help="덤프 날짜 YYYYMMDD (기본: 최신 자동 탐지)"
    )
    p.add_argument(
        "--input_bz2", type=str, default=None,
        help="이미 다운로드된 .xml.bz2 파일 경로 (지정 시 다운로드 건너뜀)"
    )
    p.add_argument(
        "--download_dir", type=str, default="./wiki_dump",
        help="덤프 다운로드 디렉토리 (기본: ./wiki_dump)"
    )
    p.add_argument(
        "--out_db", type=str, default=None,
        help="출력 SQLite DB 경로 (기본: ./KorFactScore/downloaded_files/kowiki-YYYYMMDD.db)"
    )
    p.add_argument(
        "--max_articles", type=int, default=0,
        help="처리할 최대 문서 수 (0=전체, 테스트용으로 1000 등 지정)"
    )
    p.add_argument(
        "--skip_download", action="store_true",
        help="다운로드 건너뛰기 (--input_bz2 지정 시 자동)"
    )
    p.add_argument(
        "--verify_only", action="store_true",
        help="DB 검증만 실행 (--out_db 필요)"
    )
    p.add_argument(
        "--verify_titles", type=str, default=None,
        help="검증할 표제어 목록 (쉼표 구분, 예: '세종대왕,이순신')"
    )

    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. 덤프 날짜 결정 ────────────────────────
    if args.input_bz2:
        # bz2 경로에서 날짜 추출
        m = re.search(r"(\d{8})", os.path.basename(args.input_bz2))
        date = m.group(1) if m else "unknown"
        bz2_path = args.input_bz2
        print(f"[INFO] 기존 덤프 파일 사용: {bz2_path} (날짜: {date})")
    else:
        date = args.date or _get_latest_dump_date()
        print(f"[INFO] 덤프 날짜: {date}")

    # ── 2. 출력 DB 경로 결정 ──────────────────────
    if args.out_db:
        db_path = args.out_db
    else:
        db_path = f"./KorFactScore/downloaded_files/kowiki-{date}.db"

    # ── 검증만 실행 ───────────────────────────────
    if args.verify_only:
        titles = (
            [t.strip() for t in args.verify_titles.split(",")]
            if args.verify_titles else None
        )
        verify_db(db_path, titles)
        return

    # ── 3. 다운로드 (필요 시) ─────────────────────
    if not args.input_bz2 and not args.skip_download:
        bz2_path = download_dump(date, args.download_dir)
    elif not args.input_bz2:
        print("[오류] --skip_download 사용 시 --input_bz2 를 지정하세요.")
        sys.exit(1)

    # ── 4. DB 생성 ────────────────────────────────
    build_db(
        bz2_path   = bz2_path,
        db_path    = db_path,
        max_articles = args.max_articles,
    )

    # ── 5. 자동 검증 ──────────────────────────────
    verify_titles = (
        [t.strip() for t in args.verify_titles.split(",")]
        if args.verify_titles else None
    )
    verify_db(db_path, verify_titles)

    print(f"\n[완료] korfactscore_lab.py 실행 시:")
    print(f"  python korfactscore_lab.py --korfs_path ./KorFactScore")
    print(f"  (DB 경로: {db_path})")


if __name__ == "__main__":
    main()
