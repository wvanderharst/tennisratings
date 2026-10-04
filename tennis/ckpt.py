"""Checkpoints for the sequential replays, so a daily update only replays the matches after the checkpoint date.

Every replay walks the matches in date order. At the checkpoint date (the Monday CUTOFF_WEEKS before the newest match) its
whole state is saved to state/<tag>.pkl, together with whatever the calling script collects along the way. The next run
loads it and replays only the matches from that date on; on the way it saves a new checkpoint at the new date.

A checkpoint is used only when it fits the data exactly: same code (all tennis/*.py), same settings, and every match
before its date unchanged (a hash of those rows). Anything else -> full replay from 1968 and a fresh checkpoint.
So the result is always identical to a full rebuild. TENNIS_FULL=1 ignores all checkpoints (and still writes new ones)."""
import datetime, glob, hashlib, json, os, pickle
import paths as P

DIR = os.path.join(P.ROOT, "state")
CUTOFF_WEEKS = 4          # data of the last few weeks is still being corrected/moved between files by TennisMyLife
FULL = os.environ.get("TENNIS_FULL") == "1"
_code = None
_digests = {}


def code_hash():
    global _code
    if _code is None:
        h = hashlib.sha1()
        for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*.py"))):
            h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
        _code = h.hexdigest()
    return _code


def cutoff(rows):
    """Checkpoint date: the Monday at least CUTOFF_WEEKS before the newest match."""
    d = max(r["when"] for r in rows) - datetime.timedelta(weeks=CUTOFF_WEEKS)
    return d - datetime.timedelta(days=d.weekday())


def digest(rows, before):
    """Hash of every row dated before `before` (rows are sorted by date, so this is a prefix), of the later rows of any
    tournament that already started before it (some per-event inputs look at the whole event: host country, entrants;
    rows of a running event carry their own day's date, and TennisMyLife reuses an id now and then), and of the
    birthdates of the players in those rows (ages enter the replays; the WTA birthdate file is rebuilt on every download)."""
    import confounder_harness as H
    key = (id(rows), len(rows), before, id(H.birth))
    if key not in _digests:
        h = hashlib.sha1(); players = set(); tids = set()
        for r in rows:
            if r["when"] >= before and r["tid"] not in tids: continue
            if r["when"] < before: tids.add(r["tid"])
            h.update(repr(tuple(r.items())).encode()); players.add(r["a"]); players.add(r["b"])
        h.update(repr(sorted((p, H.birth.get(p)) for p in players)).encode())
        _digests[key] = h.hexdigest()
    return _digests[key]


def first_at(rows, day):
    """Index of the first row dated on or after `day`."""
    for i, r in enumerate(rows):
        if r["when"] >= day: return i
    return len(rows)


class Checkpoint:
    """tag: file name; rows: the full sorted match list the replay is built from; params: anything else the replayed
    result depends on (settings, script-level inputs computed from the whole data set, ...), JSON- or repr-able."""

    def __init__(self, tag, rows, params=None):
        self.tag, self.rows = tag, rows
        self.key = hashlib.sha1((code_hash() + tag + repr(params)).encode()).hexdigest()
        self.cut = cutoff(rows)
        self.path = os.path.join(DIR, tag + ".pkl")

    def load(self):
        """The saved blob if it fits the current data, else None."""
        if FULL or not os.path.exists(self.path): return None
        try:
            with open(self.path, "rb") as f: ck = pickle.load(f)
        except Exception:
            return None
        if ck.get("key") != self.key or ck["cut"] > self.cut or ck["digest"] != digest(self.rows, ck["cut"]):
            print(f"  [{self.tag}] checkpoint does not match the data -> full replay", flush=True)
            return None
        print(f"  [{self.tag}] resuming from checkpoint {ck['cut']}", flush=True)
        return ck

    def save(self, blob):
        """blob: dict with the replay state at self.cut (built by the caller); written atomically."""
        os.makedirs(DIR, exist_ok=True)
        blob = dict(blob, key=self.key, cut=self.cut, digest=digest(self.rows, self.cut))
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f: pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.path)


def restore(target, saved):
    """Put saved contents into an existing container in place (hooks keep their references)."""
    if isinstance(target, dict):
        target.clear(); target.update(saved)
    elif isinstance(target, list):
        target[:] = saved
    else:
        raise TypeError(type(target))


class Loop:
    """Checkpoint for a loop over date-sorted items (e.g. events) that carries state from one item to the next
    (a random-number stream, running totals). Usage:
        L = Loop(tag, allrows, params, [date of each item])
        state = L.saved or initial state;  for k in range(L.start, n): L.at(k, state_fn); ...;  L.at(n, state_fn)"""

    def __init__(self, tag, allrows, params, dates):
        self.ck = Checkpoint(tag, allrows, params)
        self.k_cut = sum(1 for d in dates if d < self.ck.cut)
        ck = self.ck.load()
        self.saved = ck["state"] if ck else None
        self.start = ck["k"] if ck else 0

    def at(self, k, state_fn):
        if k == self.k_cut and (self.saved is None or k > self.start):
            self.ck.save(dict(k=k, state=state_fn()))
