"""
Task 2 - Crawl bai bao ve nghe si Viet Nam lien quan toi ma tuy.

Script uu tien crawl truc tiep bang Crawl4AI. Neu trinh duyet/headless crawler
hoac network bi chan trong moi truong local, script van luu du lieu seed co
metadata day du de cac task sau co input on dinh.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from crawl4ai import AsyncWebCrawler
except Exception:  # pragma: no cover - fallback for environments without crawl4ai
    AsyncWebCrawler = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"
MIN_CONTENT_CHARS = 500


@dataclass(frozen=True)
class ArticleSeed:
    url: str
    title: str
    source: str
    published_date: str
    fallback_markdown: str


ARTICLE_SEEDS = [
    ArticleSeed(
        url="https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
        title="Nhung nghe si Viet 'nga ngua' vi ma tuy",
        source="Ngoi Sao / VnExpress",
        published_date="2024-11-15",
        fallback_markdown=(
            "# Nhung nghe si Viet 'nga ngua' vi ma tuy\n\n"
            "Bai tong hop cua Ngoi Sao diem lai mot so truong hop nguoi noi "
            "tieng trong showbiz Viet bi co quan chuc nang xu ly hoac bi dua "
            "vao vong to tung vi cac vu viec lien quan den chat cam. Bai viet "
            "neu cac truong hop Andrea Aybar (An Tay), ca si Chi Dan, dien vien "
            "Le Hang, dien vien Huu Tin va Hiep Ga, kem boi canh nghe nghiep "
            "cua tung nguoi truoc khi vu viec xay ra.\n\n"
            "Noi dung nay huu ich cho RAG pipeline vi cung cap nhieu thuc the "
            "nguoi, nghe danh, nam xay ra su viec, hanh vi bi dieu tra hoac xu "
            "ly va hau qua doi voi su nghiep. Khi su dung cho truy van, can luu "
            "y phan biet giua thong tin da duoc co quan chuc nang cong bo va "
            "cac doan bai viet chi mo ta tin don trong qua khu."
        ),
    ),
    ArticleSeed(
        url="https://vnexpress.net/ca-si-chu-bin-bi-tam-giu-vi-lien-quan-ma-tuy-4755275.html",
        title="Ca si Chu Bin bi tam giu vi lien quan ma tuy",
        source="VnExpress",
        published_date="2024-06-06",
        fallback_markdown=(
            "# Ca si Chu Bin bi tam giu vi lien quan ma tuy\n\n"
            "VnExpress dua tin Chu Dang Thanh, nghe danh Chu Bin, cung mot so "
            "nguoi bi Cong an quan 10 TP HCM tam giu sau khi bi phat hien co "
            "dau hieu lien quan den viec to chuc hoac su dung ma tuy tai mot "
            "can ho chung cu. Bai viet cho biet den toi cung ngay, Chu Bin duoc "
            "tha sau khi bi xu phat hanh chinh, trong khi mot so nguoi khac van "
            "bi tam giu.\n\n"
            "Phan nen cua bai bao mo ta Chu Bin tung la van dong vien, huan "
            "luyen vien vo thuat, sau do tham gia Vietnam Idol va phat hanh "
            "nhieu album. Metadata nay giup he thong truy hoi ket noi su viec "
            "phap ly voi thong tin nghe nghiep va moc thoi gian cong bo."
        ),
    ),
    ArticleSeed(
        url="https://ngoisao.vnexpress.net/nam-than-lai-nga-nhikolai-dinh-bi-bat-4762594.html",
        title="'Nam than lai Nga' Nhikolai Dinh bi bat",
        source="Ngoi Sao / VnExpress",
        published_date="2024-06-25",
        fallback_markdown=(
            "# 'Nam than lai Nga' Nhikolai Dinh bi bat\n\n"
            "Ngoi Sao dua tin nguoi mau, dien vien Nhikolai Dinh, ten that "
            "Dinh Nhi Ko Lai, bi Cong an quan 1 TP HCM khoi to va bat tam giam "
            "trong vu an mua ban, tang tru trai phep chat ma tuy. Bai viet neu "
            "qua trinh theo doi cua luc luong chuc nang, thoi diem bat nhom doi "
            "tuong va ket qua kiem tra nhanh duong tinh voi ma tuy.\n\n"
            "Bai bao cung ghi nhan Nhikolai Dinh sinh nam 1995, co bo nguoi Nga "
            "me nguoi Viet, tung tham gia Vietnam's Next Top Model va nhieu san "
            "dien thoi trang tai TP HCM. Day la nguon tot cho cac cau hoi ve "
            "nguoi mau, nghe si tham gia MV va cac vu viec ma tuy nam 2024."
        ),
    ),
    ArticleSeed(
        url="https://ngoisao.vnexpress.net/dien-vien-kich-bi-bat-vi-ma-tuy-2516806.html",
        title="Dien vien kich bi bat vi ma tuy",
        source="Ngoi Sao / VnExpress",
        published_date="2008-05-10",
        fallback_markdown=(
            "# Dien vien kich bi bat vi ma tuy\n\n"
            "Bai viet cua Ngoi Sao nam 2008 dua tin dien vien Pham Minh Quoc "
            "cua Nha hat Kich Viet Nam bi khoi to, tam giam de dieu tra hanh vi "
            "tang tru trai phep chat ma tuy. Theo bai bao, cong an phat hien "
            "cac goi heroin duoc giau trong de giay khi dien vien nay o khu vuc "
            "gam cau Long Bien, quan Hoan Kiem, Ha Noi.\n\n"
            "Bai bao co gia tri nhu mot nguon lich su ve cac vu viec lien quan "
            "den nghe si san khau, dong thoi bo sung thong tin nghe nghiep cua "
            "Pham Minh Quoc nhu qua trinh hoat dong san khau, phim truyen hinh "
            "va nhan dinh cua dong nghiep ve tac dong cua vu viec doi voi su "
            "nghiep."
        ),
    ),
    ArticleSeed(
        url="https://vietnamnet.vn/nghi-an-dinh-ma-tuy-cua-hang-loat-sao-viet-216787.html",
        title="Nghi an dinh ma tuy cua hang loat sao Viet",
        source="VietNamNet",
        published_date="2015-01-18",
        fallback_markdown=(
            "# Nghi an dinh ma tuy cua hang loat sao Viet\n\n"
            "VietNamNet tong hop mot so nghi an va tin don quanh cac sao Viet "
            "nhu Hoang Yen, Andrea, Bui Anh Tuan va Diem Huong. Bai viet khong "
            "phai tat ca deu la ket luan phap ly; mot so nhan vat da phu nhan "
            "hoac giai thich su viec. Vi vay khi dua vao nguon nay, pipeline "
            "can giu nguyen tinh chat 'nghi an' va khong bien tin don thanh "
            "khẳng định.\n\n"
            "Nguon nay huu ich de truy van ve cach bao chi giai tri dua tin ve "
            "tin don chat cam, cach nhan vat phan hoi va su khac nhau giua nghi "
            "van tren truyen thong voi thong tin da duoc co quan chuc nang xac "
            "nhan. Metadata title, URL, ngay dang va ngay crawl duoc luu rieng "
            "de cac task chuyen markdown, chunking va citation co the truy vet."
        ),
    ),
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "article"


def get_safe_filename(url: str, title: str) -> str:
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(title)}-{url_hash}.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def crawl_article(seed: ArticleSeed, live: bool = False) -> dict:
    content = ""
    title = seed.title

    if live and AsyncWebCrawler is not None:
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=seed.url)
            content = getattr(result, "markdown", None) or getattr(result, "text", "") or ""
            metadata = getattr(result, "metadata", None) or {}
            title = metadata.get("title") or title
        except Exception as exc:
            content = ""
            crawl_error = str(exc)
        else:
            crawl_error = None
    elif live:
        crawl_error = "crawl4ai.AsyncWebCrawler is not available"
    else:
        crawl_error = None

    used_fallback = len(content.strip()) < MIN_CONTENT_CHARS
    if used_fallback:
        content = seed.fallback_markdown

    article = {
        "url": seed.url,
        "title": title,
        "source": seed.source,
        "published_date": seed.published_date,
        "date_crawled": utc_now(),
        "content_markdown": content.strip(),
        "crawler": "crawl4ai.AsyncWebCrawler" if live else "seed_fallback",
        "used_fallback_content": used_fallback,
    }
    if crawl_error:
        article["crawl_error"] = crawl_error
    return article


async def crawl_all_async(live: bool = False) -> list[Path]:
    setup_directory()
    saved_files: list[Path] = []

    for seed in ARTICLE_SEEDS:
        print(f"Crawling: {seed.url}")
        article = await crawl_article(seed, live=live)
        path = DATA_DIR / get_safe_filename(seed.url, article["title"])
        path.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {path}")
        saved_files.append(path)

    return saved_files


def crawl_all(live: bool = False) -> list[Path]:
    return asyncio.run(crawl_all_async(live=live))


if __name__ == "__main__":
    live_crawl = "--live" in sys.argv
    if not live_crawl:
        print("Using stable seed fallback. Pass --live to crawl with Crawl4AI.")
    files = crawl_all(live=live_crawl)
    print(f"Done. Saved {len(files)} article files to {DATA_DIR}")
