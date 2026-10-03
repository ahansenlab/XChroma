#!/usr/bin/env python
"""按 Epiphany 论文 Table 3 的 ENCSR accession 拉出对应的 BAM 文件 URL。

⛔ accession 逐字来自论文,不自己挑数据集 —— 对手的输入必须对齐它的官方配置。
   注意第一条轨是 DNase-seq,不是 ATAC。

输出: <POOL>/epiphany/bam_urls_<CELL>.tsv   (mark \t accession \t url)
"""

import json, sys, urllib.request

from pathlib import Path
import argparse

TABLE3 = {
    "K562":   {"DNase": "ENCSR000EOT", "CTCF": "ENCSR000DWE", "H3K27ac": "ENCSR000AKP",
               "H3K27me3": "ENCSR000AKQ", "H3K4me3": "ENCSR000DWD"},
    "HCT116": {"DNase": "ENCSR000ENM", "CTCF": "ENCSR000DTO", "H3K27ac": "ENCSR661KMA",
               "H3K27me3": "ENCSR810BDB", "H3K4me3": "ENCSR333OPW"},

    "HepG2":  {"DNase": "ENCSR000EJV", "CTCF": "ENCSR000DUG", "H3K27ac": "ENCSR000AMO",
               "H3K27me3": "ENCSR000DUE", "H3K4me3": "ENCSR000DUF"},
}
HDR = {"accept": "application/json"}


def get(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=HDR)))


def bams(acc):
    """该 experiment 下 hg38 的 alignments BAM。

    ⚠ ENCODE 对同一个实验会保留多轮处理(见 ENCSR000AKP: 2016/2020/2022 三版,
      同一批生物学重复各出现一次)。全部 merge 会把同样的 read 重复计入 2-3 遍,
      RPKM 归一化也救不回来(深度是假的)。规则: 只取 date_created 最新的那一轮,
      并报出它覆盖了哪些生物学重复,方便核对没漏 rep。
    """
    e = get(f"https://www.encodeproject.org/experiments/{acc}/?format=json")
    cand = [f for f in e["files"]
            if f.get("file_format") == "bam" and f.get("assembly") == "GRCh38"
            and f.get("output_type") == "alignments" and f.get("status") == "released"]
    if not cand:
        return [], []
    newest = max(f["date_created"][:10] for f in cand)
    keep = [f for f in cand if f["date_created"][:10] == newest]
    reps = sorted({r for f in keep for r in f.get("biological_replicates", [])})
    out = [(f["accession"],
            "https://encode-public.s3.amazonaws.com/" + f["s3_uri"].split("encode-public/")[-1]
            if f.get("s3_uri") else "https://www.encodeproject.org" + f["href"])
           for f in keep]
    return out, (newest, reps, len(cand))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cells', default='K562,HCT116,HepG2')
    args = parser.parse_args()
    if set(args.cells.split(',')) - set(TABLE3):
        parser.error('Supported cells: K562,HCT116,HepG2')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('--out must be an empty directory')
    args.out.mkdir(parents=True, exist_ok=True)
    for cell in args.cells.split(","):
        marks = TABLE3[cell]
        rows = []
        for mark, acc in marks.items():
            b, info = bams(acc)
            if not b:
                sys.exit(f"{cell} {mark} {acc}: 没找到 GRCh38 alignments BAM")
            date, reps, ncand = info
            for a, u in b:
                rows.append((mark, a, u))
            print(f"{cell:7s} {mark:9s} {acc}  取 {len(b)}/{ncand} BAM  "
                  f"(最新分析 {date}, 生物学重复 {reps})", flush=True)
        p = args.out / f"bam_urls_{cell}.tsv"
        with open(p, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(f"  -> {p}  ({len(rows)} 行)\n", flush=True)


if __name__ == '__main__':
    main()
