r"""Read individual members out of a remote ZIP without downloading it.

The Google surface-code corpus is published as ZIP archives of 5.7 GB to 65 GB.
Only a handful of members are needed to certify real hardware syndromes, and
pulling 5.7 GB to read 40 MB of them is the kind of cost that quietly turns a
milestone into a "later". ZIP is a random-access format -- the central
directory sits at the END of the file and names every member's offset -- so
three HTTP range requests are enough to list an archive, and one more per
member to fetch it.

WHAT THIS DOES NOT DO. It does not verify the archive's integrity as a whole,
because it never sees the whole archive. Each member IS checked against the
CRC-32 recorded for it in the central directory, which is what protects
against a truncated or corrupted range response. The provenance claim that
matters -- that these bytes are Google's published data -- rests on the Zenodo
DOI and the record's own checksums, and the fetched members are recorded with
their sizes and CRCs so a reader with the full archive can confirm them.
"""

from __future__ import annotations

import struct
import urllib.request
import zlib

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CEN_SIG = b"PK\x01\x02"


class RemoteZip:
    """Random access to a ZIP over HTTP range requests."""

    def __init__(self, url: str, *, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout
        self.size = self._length()
        self.entries = self._central_directory()

    # ---- transport -------------------------------------------------------
    def _get(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            if r.status != 206:
                raise RuntimeError(
                    f"server ignored the Range header (status {r.status}); "
                    f"ranged extraction is not available for this URL")
            return r.read()

    def _length(self) -> int:
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            n = r.headers.get("Content-Length")
            if not n:
                raise RuntimeError("no Content-Length; cannot seek the archive")
            return int(n)

    # ---- directory -------------------------------------------------------
    def _central_directory(self) -> dict[str, dict]:
        tail_len = min(self.size, 128 * 1024)
        tail = self._get(self.size - tail_len, self.size - 1)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise RuntimeError("no end-of-central-directory record found")
        cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
        total = struct.unpack("<H", tail[i + 10:i + 12])[0]

        # ZIP64: the 32-bit fields saturate on archives past 4 GB, and every
        # archive here is past 4 GB, so this is the normal path rather than an
        # edge case.
        j = tail.rfind(EOCD64_LOCATOR_SIG)
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or total == 0xFFFF or j >= 0:
            if j < 0:
                raise RuntimeError("ZIP64 needed but no locator present")
            eocd64_off = struct.unpack("<Q", tail[j + 8:j + 16])[0]
            head = self._get(eocd64_off, eocd64_off + 55)
            if head[:4] != EOCD64_SIG:
                raise RuntimeError("ZIP64 end-of-central-directory not at the "
                                   "offset the locator gives")
            total = struct.unpack("<Q", head[32:40])[0]
            cd_size = struct.unpack("<Q", head[40:48])[0]
            cd_off = struct.unpack("<Q", head[48:56])[0]

        cd = self._get(cd_off, cd_off + cd_size - 1)
        out: dict[str, dict] = {}
        p = 0
        for _ in range(total):
            if cd[p:p + 4] != CEN_SIG:
                break
            (method, _t, _d, crc, csize, usize, nlen, elen, clen,
             _dsk, _ia, _ea, lho) = struct.unpack("<HHHIIIHHHHHII", cd[p + 10:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
            # ZIP64 extra field overrides the saturated 32-bit values, in a
            # fixed ORDER but only for the fields that actually saturated --
            # reading them positionally without checking which are present is
            # the classic way to get a plausible, wrong offset.
            if 0xFFFFFFFF in (csize, usize, lho):
                q = 0
                while q + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                    if hid == 0x0001:
                        blk = extra[q + 4:q + 4 + hsz]
                        k = 0
                        if usize == 0xFFFFFFFF and k + 8 <= len(blk):
                            usize = struct.unpack("<Q", blk[k:k + 8])[0]
                            k += 8
                        if csize == 0xFFFFFFFF and k + 8 <= len(blk):
                            csize = struct.unpack("<Q", blk[k:k + 8])[0]
                            k += 8
                        if lho == 0xFFFFFFFF and k + 8 <= len(blk):
                            lho = struct.unpack("<Q", blk[k:k + 8])[0]
                            k += 8
                        break
                    q += 4 + hsz
            out[name] = {"method": method, "crc": crc, "csize": csize,
                         "usize": usize, "header_offset": lho}
            p += 46 + nlen + elen + clen
        return out

    # ---- members ---------------------------------------------------------
    def read(self, name: str) -> bytes:
        e = self.entries[name]
        lh = self._get(e["header_offset"], e["header_offset"] + 29)
        if lh[:4] != b"PK\x03\x04":
            raise RuntimeError(f"{name}: no local header at the recorded offset")
        nlen, elen = struct.unpack("<HH", lh[26:30])
        start = e["header_offset"] + 30 + nlen + elen
        raw = self._get(start, start + e["csize"] - 1)
        if e["method"] == 0:
            data = raw
        elif e["method"] == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise RuntimeError(f"{name}: unsupported compression {e['method']}")
        # A TRUNCATED RANGE RESPONSE LOOKS LIKE VALID DATA. The CRC from the
        # central directory is the only thing that distinguishes them, so it
        # is checked here rather than trusted to the caller.
        if zlib.crc32(data) & 0xFFFFFFFF != e["crc"]:
            raise RuntimeError(f"{name}: CRC mismatch -- the fetched bytes are "
                               f"not the member the directory describes")
        return data


if __name__ == "__main__":
    import sys
    z = RemoteZip(sys.argv[1])
    pat = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"{len(z.entries)} members, archive {z.size/1e9:.2f} GB")
    n = 0
    for k, v in z.entries.items():
        if pat in k:
            print(f"  {k}  {v['usize']/1e6:.2f} MB")
            n += 1
            if n >= 40:
                print("  ...")
                break
