"""
score_driven.py — two alternatives to what the simulator currently uses.

Glicko replaces Elo. Instead of a single number it carries a rating deviation: how
unsure we are. An uncertain player moves further after a result, and a result against
an uncertain opponent counts for less. Deviation grows back during time off, which is
the part plain Elo has no way to express.

GAS (generalized autoregressive score) replaces the hand-built EWMA. The rates become
time-varying parameters of a binomial model and each update is the score of the
likelihood — the derivative of the log-likelihood with respect to the parameter —
scaled and multiplied by a learning rate estimated by maximum likelihood. Following
Gorgi, Koopman and Lit (2019), "The analysis and forecasting of tennis matches by using
a high dimensional dynamic model", JRSS-A 182(4).

Two things fall out of that formulation rather than being imposed by hand:

  * The update is proportional to (observed wins - expected wins), so a 155-point match
    moves a rating three times as far as a 45-point one. The EWMA weighted both the same.
  * One observation moves the server's attack parameter up and the returner's defence
    parameter down by the same amount, because the score with respect to one is the
    negative of the score with respect to the other. The half-and-half credit split I
    imposed by hand is what the derivative says to do.
"""

import csv
import glob
import math
import os
from collections import defaultdict
from datetime import date

import numpy as np
from scipy.optimize import minimize

import build_ratings as B

# ---------------------------------------------------------------------------
# Match counts
# ---------------------------------------------------------------------------

def _counts(row, me, them):
    """Raw point counts for one side. GAS needs the counts, not the rates."""
    def n(prefix, field):
        try:
            return int(row[prefix + "_" + field])
        except (ValueError, TypeError, KeyError):
            return None

    sv, f_in, f_won, s_won, dfs = (n(me, x) for x in ("svpt", "1stIn", "1stWon", "2ndWon", "df"))
    if any(x is None for x in (sv, f_in, f_won, s_won, dfs)):
        return None
    s_pts = sv - f_in
    s_in = s_pts - dfs
    if min(sv, f_in, s_pts, s_in) <= 0:
        return None
    if f_won > f_in or s_won > s_in:
        return None
    out = {"sv": sv, "f_in": f_in, "f_won": f_won, "s_pts": s_pts, "s_in": s_in, "s_won": s_won}
    try:
        games = int(row[me + "_SvGms"])
        breaks = int(row[me + "_bpFaced"]) - int(row[me + "_bpSaved"])
        if games > 0 and 0 <= breaks <= games:
            out["sv_gms"] = games
            out["holds"] = games - breaks
    except (ValueError, TypeError, KeyError):
        pass
    return out


parse_date = B.parse_date

SET_RE = __import__("re").compile(r"^(\d+)-(\d+)")


def parse_close(score, best_of):
    """Tiebreaks played and won by the winner, and how a deciding set went.

    Returns (tb_played, tb_won_by_winner, decider) where decider is 1 if the match went
    to a final set and the first-named player won it, 0 if it went to a final set and he
    lost it, and None if it never got there.
    """
    if not score:
        return 0, 0, None
    upper = score.upper()
    if "RET" in upper or "W/O" in upper or "DEF" in upper:
        return 0, 0, None
    sets = []
    for token in score.split():
        m = SET_RE.match(token)
        if m:
            sets.append((int(m.group(1)), int(m.group(2))))
    tb_played = tb_won = 0
    for a, b in sets:
        if {a, b} == {6, 7}:                  # 7-6 or 6-7 is a tiebreak set
            tb_played += 1
            tb_won += (a > b)
    need = 2 if best_of == 3 else 3
    won = sum(1 for a, b in sets if a > b)
    lost = len(sets) - won
    decider = 1 if (won == need and lost == need - 1) else None
    return tb_played, tb_won, decider


def parse_sets(score):
    """Sets won by the winner and the loser. Returns None for anything that did not
    finish on court — retirements and walkovers say nothing about relative strength."""
    if not score:
        return None
    upper = score.upper()
    if "RET" in upper or "W/O" in upper or "DEF" in upper or "ABD" in upper or "UNFINISHED" in upper:
        return None
    w = l = 0
    for token in score.split():
        m = SET_RE.match(token)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            w += 1
        elif b > a:
            l += 1
    if w + l < 2 or w <= l:
        return None
    return w, l


# tourney_level tells us what tier a match belongs to, which is more reliable than
# guessing from a filename — the website and the old GitHub repo name their files
# differently, and both may be present.
LEVEL_OF = {"C": "chall", "S": "chall", "Q": "qual"}


def csv_sources(outdir):
    """Every match CSV we can find, whatever layout the data arrived in.

    Preferred: a tml-data/ folder downloaded from stats.tennismylife.org, which carries
    the full Challenger tour and qualifying rather than the tour-level-only extract the
    frozen GitHub repo holds. Falls back to the older tml/ and chall/ folders.
    """
    for folder in ("tml-data", "tml", "chall"):
        root = os.path.join(outdir, folder)
        if not os.path.isdir(root):
            continue
        for path in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
            yield path


def load_counts(outdir, levels=("tour", "chall", "qual")):
    """Every match as (date, id, A, counts_A, B, counts_B, surface, level, best_of,
    sets, close), in date order and de-duplicated."""
    out, seen = [], set()
    for path in csv_sources(outdir):
        for row in csv.DictReader(open(path, encoding="utf-8", errors="replace")):
            when = parse_date(row.get("tourney_date"))
            if not when:
                continue
            a = (row.get("winner_name") or "").strip()
            b = (row.get("loser_name") or "").strip()
            if not a or not b or a == b:
                continue

            level = LEVEL_OF.get((row.get("tourney_level") or "").strip().upper(), "tour")
            if level not in levels:
                continue

            # De-duplicate on the match itself rather than its identifiers. The same
            # match can arrive from two sources under different tourney_ids — the
            # website's Challenger files carry a handful of old events renumbered under
            # a recent id — so date, both names and the scoreline is the safer key.
            key = (when, a, b, (row.get("score") or "").strip())
            if key in seen:
                continue

            ca, cb = _counts(row, "w", "l"), _counts(row, "l", "w")
            if not ca or not cb:
                continue
            seen.add(key)

            surface = row.get("surface", "")
            if surface not in ("Hard", "Clay", "Grass", "Carpet"):
                surface = ""
            try:
                bo = int(row["best_of"])
            except (ValueError, TypeError, KeyError):
                bo = 3
            mid = f"{level[:2]}{row.get('tourney_id','')}-{row.get('match_num','')}"
            out.append((when, mid, a, ca, b, cb, surface, level, bo,
                        parse_sets(row.get("score")),
                        parse_close(row.get("score"), bo)))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


# ---------------------------------------------------------------------------
# Glicko
# ---------------------------------------------------------------------------

Q = math.log(10) / 400.0


class Glicko:
    """Glicko-1, updated one match at a time rather than in rating periods.

    r    the rating, same scale as Elo
    rd   the rating deviation — how far the truth might be from r
    c    how fast deviation grows back per day of inactivity
    """

    def __init__(self, r0=1500.0, rd0=350.0, rd_min=30.0, c=0.6):
        self.r0, self.rd0, self.rd_min, self.c = r0, rd0, rd_min, c
        self.r = defaultdict(lambda: r0)
        self.rd = defaultdict(lambda: rd0)
        self.seen = {}

    def _current_rd(self, p, today):
        rd = self.rd[p]
        last = self.seen.get(p)
        if last is not None:
            gap = (today - last).days
            if gap > 0:
                rd = min(self.rd0, math.sqrt(rd * rd + self.c * self.c * gap))
        return max(rd, self.rd_min)

    @staticmethod
    def _g(rd):
        return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / (math.pi ** 2))

    def expected(self, a, b, today):
        """Probability A beats B, widening the curve by both players' uncertainty."""
        rd_a, rd_b = self._current_rd(a, today), self._current_rd(b, today)
        g = self._g(math.sqrt(rd_a ** 2 + rd_b ** 2))
        return 1.0 / (1.0 + 10 ** (-g * (self.r[a] - self.r[b]) / 400.0))

    def update(self, winner, loser, today):
        ra, rb = self.r[winner], self.r[loser]
        rda, rdb = self._current_rd(winner, today), self._current_rd(loser, today)

        for me, opp, r_me, r_opp, rd_me, rd_opp, s in (
                (winner, loser, ra, rb, rda, rdb, 1.0),
                (loser, winner, rb, ra, rdb, rda, 0.0)):
            g = self._g(rd_opp)
            e = 1.0 / (1.0 + 10 ** (-g * (r_me - r_opp) / 400.0))
            d2 = 1.0 / (Q * Q * g * g * e * (1 - e))
            denom = 1.0 / (rd_me ** 2) + 1.0 / d2
            self.r[me] = r_me + (Q / denom) * g * (s - e)
            self.rd[me] = max(self.rd_min, math.sqrt(1.0 / denom))
            self.seen[me] = today


# ---------------------------------------------------------------------------
# GAS
# ---------------------------------------------------------------------------

def logit(p):
    return math.log(p / (1 - p))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


# Each group is one binomial observation per player per match.
#   trials, successes: which counts to read
#   paired: whether the probability depends on the opponent as well
GROUPS = {
    "f1": {"n": "sv",    "y": "f_in",  "paired": False},   # first serves landed
    "f2": {"n": "s_pts", "y": "s_in",  "paired": False},   # second serves landed
    "w1": {"n": "f_in",  "y": "f_won", "paired": True},    # points won behind a first serve
    "w2": {"n": "s_in",  "y": "s_won", "paired": True},    # ... behind a second serve
}


# How fast a rating should move is a signal-to-noise question, not a temperament one.
# The optimal gain on a new observation is signal variance over signal plus noise: a
# player whose true level is genuinely moving should be tracked quickly, and a player
# who is merely erratic around a stable level should be tracked slowly, because their
# surprises are mostly noise. Those pull in opposite directions, which is why "streaky
# players should update faster" is the wrong instinct — streakiness is the noise term.
#
# Two things do warrant a faster gain, and they are both about how much is unknown
# rather than about the player:
#
#   NEW PLAYERS. The residual autocorrelation over a player's first 25 matches is
#   +0.104, against about +0.036 for the rest of a career. The filter is demonstrably
#   too slow there, and it shows up as bias: over the first ten matches the model
#   over-predicts serve points by 2.7 percentage points, because it is still dragging
#   a newcomer toward the novice prior after they have shown they are better than that.
#
#   VETERANS. The autocorrelation ticks back up to +0.054 past 300 matches, which is
#   decline the filter has not kept up with.
#
# What does NOT warrant its own rate is the player. A player's own under-reaction is
# only 0.24 repeatable across halves of their career — an implied true spread of 0.036
# on a mean of 0.04, so barely distinguishable from zero — and it correlates +0.20 with
# their level dispersion, the wrong sign for a tracking story. If erratic players truly
# needed faster updates that correlation would be negative, because the optimal gain
# falls as noise rises. A positive one instead says both quantities are picking up the
# same thing: players whose results are hard to predict look under-reacted to whatever
# rate you use, and speeding them up would chase noise.
#
# Elo has had this for years in the K = 250/(n+5)^0.4 decay. The score-driven models
# here run a single learning rate for everyone at every stage, and that is the gap.
# Fitted on held-out matches. Doubling the rate over a player's first fifteen or so
# matches is the whole of the gain; tripling it overshoots, and stretching the decay
# past forty matches gives it back. So the correction is small, early, and short.
GAIN_EARLY = 2.0          # multiplier on the learning rate for a player's first matches
GAIN_SETTLE = 15.0        # matches over which it decays back to one


def gain(n, early=GAIN_EARLY, settle=GAIN_SETTLE):
    """Learning-rate multiplier for a player with n matches behind them."""
    return 1.0 + (early - 1.0) * math.exp(-n / settle)


def filter_group(matches, key, a_rate, b_persist, scaling, base, until=None,
                 collect=None):
    """Run the score-driven filter for one group and return its predictive log-likelihood.

    Parameters
        a_rate     the learning rate: how far one match moves a parameter
        b_persist  1.0 is a random walk; below 1 pulls back toward the starting level
        scaling    0 leaves the raw score, 0.5 divides by the standard deviation of the
                   score, 1.0 is the full Newton step
        base       starting value on the logit scale
    `collect`, if given, is called with (match, pre_attack, pre_defence) before each
    update, which is how the backtest reads out one-step-ahead predictions.
    """
    spec = GROUPS[key]
    paired = spec["paired"]
    att = defaultdict(lambda: base)
    dfn = defaultdict(float)
    ll = 0.0

    for m in matches:
        when, _mid, a, ca, b, cb, _surf, _lvl, _bo, _sets, _close = m
        if collect is not None:
            collect(m, att, dfn)
        if until is not None and when >= until:
            continue

        pre_att = {a: att[a], b: att[b]}
        pre_dfn = {a: dfn[a], b: dfn[b]}
        for me, cs, opp in ((a, ca, b), (b, cb, a)):
            n, y = cs[spec["n"]], cs[spec["y"]]
            if n <= 0:
                continue
            theta = pre_att[me] - (pre_dfn[opp] if paired else 0.0)
            p = sigmoid(theta)
            p = min(max(p, 1e-6), 1 - 1e-6)
            ll += y * math.log(p) + (n - y) * math.log(1 - p)

            score = y - n * p                      # d loglik / d theta
            info = n * p * (1 - p)                 # Fisher information
            if scaling:
                score /= max(info, 1e-6) ** scaling
            step = a_rate * score

            att[me] = base + b_persist * (att[me] - base) + step
            if paired:
                # the same observation says the opposite about the opponent
                dfn[opp] = b_persist * dfn[opp] - step
    return ll


def fit_group(matches, key, base, until, scalings=(0.5, 1.0)):
    """Maximum likelihood for the learning rate and persistence of one group."""
    best = None
    for scaling in scalings:
        def negll(par):
            a_rate = math.exp(par[0])
            b_persist = 1.0 / (1.0 + math.exp(-par[1]))
            if a_rate <= 0 or not math.isfinite(a_rate):
                return 1e18
            try:
                return -filter_group(matches, key, a_rate, b_persist, scaling, base, until)
            except (OverflowError, ValueError):
                return 1e18

        start = [math.log(0.02 if scaling < 1 else 0.6), 6.0]
        res = minimize(negll, start, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 5.0, "maxiter": 60})
        cand = (res.fun, math.exp(res.x[0]), 1.0 / (1.0 + math.exp(-res.x[1])), scaling)
        if best is None or cand[0] < best[0]:
            best = cand
    return {"negll": best[0], "a": best[1], "b": best[2], "scaling": best[3]}


def pool_rates(matches):
    """Overall success rate for each group, used as the starting level."""
    out = {}
    for key, spec in GROUPS.items():
        n = sum(c[spec["n"]] for m in matches for c in (m[3], m[5]))
        y = sum(c[spec["y"]] for m in matches for c in (m[3], m[5]))
        out[key] = y / n
    return out


def run_all(matches, fits, pool, callback=None):
    """Run all four groups in one pass, so a caller can read every parameter a match
    needs at the same moment. `callback(match, params)` fires before each update, where
    params maps a player name to their six numbers on the probability scale."""
    base = {k: logit(pool[k]) for k in GROUPS}
    att = {k: defaultdict(lambda b=base[k]: b) for k in GROUPS}
    dfn = {k: defaultdict(float) for k in GROUPS}

    for m in matches:
        _when, _mid, a, ca, b, cb, _surf, _lvl, _bo, _sets, _close = m
        if callback is not None:
            snap = {}
            for who in (a, b):
                other = b if who is a else a
                snap[who] = {
                    "f1": sigmoid(att["f1"][who]),
                    "f2": sigmoid(att["f2"][who]),
                    "w1": sigmoid(att["w1"][who] - dfn["w1"][other]),
                    "w2": sigmoid(att["w2"][who] - dfn["w2"][other]),
                    "a1": att["w1"][who], "d1": dfn["w1"][who],
                    "a2": att["w2"][who], "d2": dfn["w2"][who],
                }
            callback(m, snap)

        for key, spec in GROUPS.items():
            f = fits[key]
            a_rate, b_persist, scaling = f["a"], f["b"], f["scaling"]
            paired = spec["paired"]
            pre_att = {a: att[key][a], b: att[key][b]}
            pre_dfn = {a: dfn[key][a], b: dfn[key][b]}
            for me, cs, opp in ((a, ca, b), (b, cb, a)):
                n, y = cs[spec["n"]], cs[spec["y"]]
                if n <= 0:
                    continue
                theta = pre_att[me] - (pre_dfn[opp] if paired else 0.0)
                p = min(max(sigmoid(theta), 1e-6), 1 - 1e-6)
                score = y - n * p
                if scaling:
                    score /= max(n * p * (1 - p), 1e-6) ** scaling
                step = a_rate * score
                att[key][me] = base[key] + b_persist * (att[key][me] - base[key]) + step
                if paired:
                    dfn[key][opp] = b_persist * dfn[key][opp] - step
    return att, dfn


def gas_serve_point(mine, theirs):
    """Probability the server wins a point, straight from the GAS parameters."""
    w1 = sigmoid(mine["a1"] - theirs["d1"])
    w2 = sigmoid(mine["a2"] - theirs["d2"])
    return mine["f1"] * w1 + (1 - mine["f1"]) * mine["f2"] * w2


# ---------------------------------------------------------------------------
# GAS straight on the result
# ---------------------------------------------------------------------------
#
# The models above rate players on how points go and then build a match probability
# out of them, which needs an assumption about how points combine. These two skip that:
# they rate players on the thing being predicted.
#
#   OUTCOME  one Bernoulli per match. p = sigmoid(theta_i - theta_j). This is a
#            score-driven Bradley-Terry model, the Gorgi-Koopman-Lit setup, and it is
#            Elo with the K factor estimated rather than assumed.
#   SETS     one binomial per match: sets won out of sets played, with a single-set
#            probability q = sigmoid(phi_i - phi_j). Match probability then comes from
#            treating sets as independent, which is a far milder assumption than
#            treating points as independent.


def match_from_set_prob(q, best_of):
    """Probability of taking a best-of-N given a per-set probability."""
    need = 2 if best_of == 3 else 3
    total = 0.0
    for lost in range(need):
        total += math.comb(need + lost - 1, lost) * q ** need * (1 - q) ** lost
    return total


def filter_outcome(matches, a_rate, b_persist, scaling, until=None, callback=None):
    """Score-driven Bradley-Terry on the win/loss result."""
    theta = defaultdict(float)
    ll = 0.0
    for m in matches:
        when, _mid, a, _ca, b, _cb, _surf, _lvl, _bo, _sets, _close = m
        p = min(max(sigmoid(theta[a] - theta[b]), 1e-9), 1 - 1e-9)
        if callback is not None:
            callback(m, p, theta)
        if until is not None and when >= until:
            continue
        ll += math.log(p)                      # the first-named player always won
        score = 1.0 - p                        # d loglik / d theta_a
        if scaling:
            score /= max(p * (1 - p), 1e-6) ** scaling
        step = a_rate * score
        theta[a] = b_persist * theta[a] + step
        theta[b] = b_persist * theta[b] - step
    return ll


def filter_sets(matches, a_rate, b_persist, scaling, until=None, callback=None):
    """Score-driven binomial on sets won, aggregated to a match probability."""
    phi = defaultdict(float)
    ll = 0.0
    for m in matches:
        when, _mid, a, _ca, b, _cb, _surf, _lvl, bo, sets, _close = m
        q = min(max(sigmoid(phi[a] - phi[b]), 1e-9), 1 - 1e-9)
        if callback is not None:
            callback(m, match_from_set_prob(q, bo), q, phi)
        if until is not None and when >= until:
            continue
        if not sets:
            continue
        won, lost = sets
        n = won + lost
        ll += won * math.log(q) + lost * math.log(1 - q)
        score = won - n * q
        if scaling:
            score /= max(n * q * (1 - q), 1e-6) ** scaling
        step = a_rate * score
        phi[a] = b_persist * phi[a] + step
        phi[b] = b_persist * phi[b] - step
    return ll


def fit_simple(matches, which, until, scalings=(0.0, 0.5, 1.0)):
    """Maximum likelihood for the learning rate and persistence of either result model."""
    f = filter_outcome if which == "outcome" else filter_sets
    best = None
    for scaling in scalings:
        def negll(par):
            a_rate = math.exp(par[0])
            b_persist = 1.0 / (1.0 + math.exp(-par[1]))
            try:
                return -f(matches, a_rate, b_persist, scaling, until)
            except (OverflowError, ValueError):
                return 1e18

        start = [math.log({0.0: 0.08, 0.5: 0.04, 1.0: 0.02}[scaling]), 6.0]
        res = minimize(negll, start, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 2.0, "maxiter": 60})
        cand = {"negll": res.fun, "a": math.exp(res.x[0]),
                "b": 1.0 / (1.0 + math.exp(-res.x[1])), "scaling": scaling}
        if best is None or cand["negll"] < best["negll"]:
            best = cand
    return best


# ---------------------------------------------------------------------------
# The game-level model
# ---------------------------------------------------------------------------
#
# The point model builds a match probability through three independence assumptions:
# points combine into games, games into sets, sets into matches. The first is the worst
# offender — a four-point edge on serve becomes an enormous hold edge once compounded
# over a game, which is why the chained probabilities came out so extreme.
#
# This model deletes that layer. Holds are observed directly (see _counts), so hold
# probability becomes a paired parameter in its own right rather than something derived
# from points. Tiebreaks get the same treatment, from the tiebreak sets in the score
# line. Only the set and match layers are still assembled by assumption.

GAME_GROUPS = {
    "hold": {"n": "sv_gms", "y": "holds"},      # service games held
}


def filter_games(matches, a_rate, b_persist, scaling, base, until=None):
    """Score-driven paired model for holding serve. One parameter for holding, one for
    breaking, moving in opposite directions off the same observation."""
    att = defaultdict(lambda: base)
    dfn = defaultdict(float)
    ll = 0.0
    for m in matches:
        when, _mid, a, ca, b, cb, _surf, _lvl, _bo, _sets, _close = m
        if "holds" not in ca or "holds" not in cb:
            continue
        pre_a = {a: att[a], b: att[b]}
        pre_d = {a: dfn[a], b: dfn[b]}
        for me, cs, opp in ((a, ca, b), (b, cb, a)):
            n, y = cs["sv_gms"], cs["holds"]
            if n <= 0:
                continue
            p = min(max(sigmoid(pre_a[me] - pre_d[opp]), 1e-6), 1 - 1e-6)
            if until is None or when < until:
                ll += y * math.log(p) + (n - y) * math.log(1 - p)
            score = y - n * p
            if scaling:
                score /= max(n * p * (1 - p), 1e-6) ** scaling
            step = a_rate * score
            att[me] = base + b_persist * (att[me] - base) + step
            dfn[opp] = b_persist * dfn[opp] - step
    return ll


def filter_tiebreaks(matches, a_rate, b_persist, scaling, until=None):
    """Same idea for tiebreak sets, which the game model cannot reach."""
    theta = defaultdict(float)
    ll = 0.0
    for m in matches:
        when, _mid, a, _ca, b, _cb, _surf, _lvl, _bo, _sets, close = m
        n, y, _dec = close
        if n <= 0:
            continue
        p = min(max(sigmoid(theta[a] - theta[b]), 1e-6), 1 - 1e-6)
        if until is None or when < until:
            ll += y * math.log(p) + (n - y) * math.log(1 - p)
        score = y - n * p
        if scaling:
            score /= max(n * p * (1 - p), 1e-6) ** scaling
        step = a_rate * score
        theta[a] = b_persist * theta[a] + step
        theta[b] = b_persist * theta[b] - step
    return ll


def fit_games(matches, which, until, base=0.0, scalings=(0.0, 0.5, 1.0)):
    f = filter_games if which == "hold" else filter_tiebreaks
    best = None
    for scaling in scalings:
        def negll(par):
            a_rate = math.exp(par[0])
            b_persist = 1.0 / (1.0 + math.exp(-par[1]))
            try:
                if which == "hold":
                    return -f(matches, a_rate, b_persist, scaling, base, until)
                return -f(matches, a_rate, b_persist, scaling, until)
            except (OverflowError, ValueError):
                return 1e18
        start = [math.log({0.0: 0.03, 0.5: 0.02, 1.0: 0.01}[scaling]), 6.0]
        res = minimize(negll, start, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1.0, "maxiter": 60})
        cand = {"negll": res.fun, "a": math.exp(res.x[0]),
                "b": 1.0 / (1.0 + math.exp(-res.x[1])), "scaling": scaling}
        if best is None or cand["negll"] < best["negll"]:
            best = cand
    return best


def run_games(matches, hold_fit, tb_fit, base, callback=None):
    """Both game-level filters in one pass, handing parameters over before each match."""
    hold_a = defaultdict(lambda: base)
    hold_d = defaultdict(float)
    tb = defaultdict(float)

    for m in matches:
        _when, _mid, a, ca, b, cb, _surf, _lvl, _bo, _sets, close = m
        if callback is not None:
            callback(m, {a: (hold_a[a], hold_d[a], tb[a]),
                         b: (hold_a[b], hold_d[b], tb[b])})

        if "holds" in ca and "holds" in cb:
            f = hold_fit
            pre_a = {a: hold_a[a], b: hold_a[b]}
            pre_d = {a: hold_d[a], b: hold_d[b]}
            for me, cs, opp in ((a, ca, b), (b, cb, a)):
                n, y = cs["sv_gms"], cs["holds"]
                if n <= 0:
                    continue
                p = min(max(sigmoid(pre_a[me] - pre_d[opp]), 1e-6), 1 - 1e-6)
                score = y - n * p
                if f["scaling"]:
                    score /= max(n * p * (1 - p), 1e-6) ** f["scaling"]
                step = f["a"] * score
                hold_a[me] = base + f["b"] * (hold_a[me] - base) + step
                hold_d[opp] = f["b"] * hold_d[opp] - step

        n, y, _dec = close
        if n > 0:
            f = tb_fit
            p = min(max(sigmoid(tb[a] - tb[b]), 1e-6), 1 - 1e-6)
            score = y - n * p
            if f["scaling"]:
                score /= max(n * p * (1 - p), 1e-6) ** f["scaling"]
            step = f["a"] * score
            tb[a] = f["b"] * tb[a] + step
            tb[b] = f["b"] * tb[b] - step
    return hold_a, hold_d, tb


# ---------------------------------------------------------------------------
# Streakiness
# ---------------------------------------------------------------------------
#
# Two different things get called "streaky" and they need separating.
#
# MOMENTUM — good days clustering together. The standardised serve-point residual is
# autocorrelated within a player at r = +0.046 one match back, +0.034 two back, and
# nothing by five. Real, but tiny, and most of it is a same-event effect: consecutive
# matches inside one week correlate at +0.031 against +0.019 for matches more than a
# week apart. That is conditions and confidence over a few days, not a lasting state,
# and a filter with a 0.07 learning rate already absorbs almost all of it.
#
# DISPERSION — some players simply vary more, week to week, around whatever their
# rating says. This one is real and worth carrying. Split-half reliability of a
# player's residual spread is 0.43, and the reliable part is about 4% of the mean.
# It has a mechanism too: dispersion correlates -0.33 with return strength. Players
# who live on serve swing further, because their whole match hinges on one shot going
# in; players who grind from the baseline float less.
#
# The estimate for one player is noisy, so it is shrunk halfway to the field average.
# That weight was chosen on held-out matches — halfway beat both the pooled value and
# the unshrunk per-player estimate.

STREAK_SHRINK = 0.5
STREAK_MIN_MATCHES = 40


def player_dispersion(residual_sd, pooled_sd, base_scale=0.173, shrink=STREAK_SHRINK):
    """Scale of a player's form distribution, in logit units.

    residual_sd is the standard deviation of that player's standardised serve-point
    residuals; pooled_sd is the same figure across the whole field.
    """
    raw = base_scale * residual_sd / pooled_sd
    return (1 - shrink) * base_scale + shrink * raw


# ---------------------------------------------------------------------------
# Set-strength GAS, the Gorgi-Koopman-Lit specification
# ---------------------------------------------------------------------------
#
# filter_sets above puts a binomial straight on sets won, which treats a 6-4 6-4 win and
# a 4-6 6-4 6-4 win as two-from-two versus two-from-three and learns from the difference.
# That is not what the paper does.
#
# Gorgi, Koopman and Lit (2019) instead carry a *set* strength, derive the match
# probability from it by treating sets within a match as independent, and evaluate the
# likelihood on the match result alone. Only who won is observed; how many sets it took
# is not in the likelihood. The state still updates in set units, so the parameter keeps
# its interpretation, but nothing is learned from the scoreline.
#
# The score is the derivative of the match log-likelihood with respect to the state,
# taken through the chain: match probability -> set probability -> logit.
#
#     dM/dq  for a best of three is  6q - 6q^2   ... = 6q(1-q)
#     dq/dθ  is                      q(1-q)
#     score  is  (1/M) dM/dq dq/dθ   for a win
#
# It is worth keeping both versions because they answer different questions: this one
# asks what a match result tells you, the binomial version asks what a scoreline does.


def match_from_sets(q, best_of):
    """Probability of taking the match, given a per-set probability."""
    need = 2 if best_of == 3 else 3
    return sum(math.comb(need + lost - 1, lost) * q ** need * (1 - q) ** lost
               for lost in range(need))


def dmatch_dq(q, best_of):
    """Derivative of the above with respect to q."""
    need = 2 if best_of == 3 else 3
    total = 0.0
    for lost in range(need):
        c = math.comb(need + lost - 1, lost)
        total += c * (need * q ** (need - 1) * (1 - q) ** lost
                      - (lost * q ** need * (1 - q) ** (lost - 1) if lost else 0.0))
    return total


def filter_set_strength(matches, a_rate, b_persist, scaling, until=None, callback=None):
    """Score-driven set strength with the likelihood on the match result."""
    theta = defaultdict(float)
    ll = 0.0
    for m in matches:
        when, _mid, a, _ca, b, _cb, _surf, _lvl, bo, sets, _close = m
        if sets is None:                      # retirement or walkover — says nothing
            if callback is not None:
                callback(m, None, theta)
            continue

        q = min(max(sigmoid(theta[a] - theta[b]), 1e-6), 1 - 1e-6)
        pm = min(max(match_from_sets(q, bo), 1e-9), 1 - 1e-9)
        if callback is not None:
            callback(m, pm, theta)
        if until is not None and when >= until:
            continue

        ll += math.log(pm)                    # the first-named player always won
        # d log M / d theta, through the set probability
        score = dmatch_dq(q, bo) * q * (1 - q) / pm
        if scaling:
            # Fisher information for a Bernoulli match outcome, in state units
            grad = dmatch_dq(q, bo) * q * (1 - q)
            info = grad * grad / (pm * (1 - pm))
            score /= max(info, 1e-9) ** scaling
        step = a_rate * score
        theta[a] = b_persist * theta[a] + step
        theta[b] = b_persist * theta[b] - step
    return ll


# The diversity problem, which is the largest thing still wrong with the simulator.
#
# Across 142 Grand Slams and 312 Masters events since 1991, the champion is drawn from a
# far smaller pool than the model produces. Since 2010 the winner of a slam has been the
# top-rated entrant 52% of the time, one of the top two 76%, one of the top four 94%, and
# one of the top eight in every single one of sixty-six events. The simulator gives the
# top eight 65%. Roughly a third of simulated slams are won by somebody who, in thirty-five
# years of real tennis, has never won one.
#
# The draws are not the explanation: a real slam field runs 293 Elo points from top seed to
# eighth best and 566 to the median, against 378 and 591 in the simulated one, so if
# anything the simulated field is the more top-heavy.
#
# It is a tension between two things that are each independently well fitted. The set-score
# distribution needs the match-up term — without it the model produces 1.34% 6-0 sets
# against a real 2.25% — and that same term is what lets an outsider through seven rounds.
# Restricting the correlation test to closely-rated pairs, where one player being better
# cannot drive it, the first-set-to-second-set correlation is +0.045 rather than the +0.071
# measured across all matches, which allows about a third of the shock to sit at set level
# rather than match level. That would keep the scorelines and cost the outsiders something,
# but it does not close a gap this size on its own.
#
# The remaining suspicion is that real draws are not random with respect to form: a player
# arriving injured withdraws rather than losing in the first round, and the seeding protects
# the best players from each other in a way the model reproduces structurally but not in the
# selection of who actually turns up. That is not something match statistics can settle.


# Whether an elite returner gains more against an elite server than the paired form allows.
# The intuition is reasonable: if breaking is what decides matches at the top, a player who
# breaks big servers should be worth more than a simple difference of ratings suggests.
# The data says the paired form is very nearly right.
#
# Regressing the observed hold logit on the server's own hold rating, the returner's break
# rating, and their interaction, across 59,551 player-matches since 2020:
#
#     server's hold rating      +1.061
#     returner's break rating   -0.877
#     interaction               -0.047
#
# A paired model assumes +1, -1 and zero. The interaction is small and in the direction the
# intuition predicts — a good returner does gain slightly more against a good server — but
# at -0.047 it is a twentieth of the main effects, and modelling it would move a match by a
# fraction of a percentage point.
#
# The apparent effect in raw rates is a scale illusion. Against elite servers Sinner breaks
# 21.7% where the tour breaks 13.1%, a gap of 8.6 points; against weak servers he breaks
# 42.2% where the tour breaks 26.9%, a gap of 15.3. That looks like his edge shrinking
# against the best, but on the logit scale the two are the same edge — 0.61 and 0.68 — and
# the difference is only that a percentage point near 13% is a bigger logit step than one
# near 27%. The returner's break rating coefficient of -0.877 rather than -1 says the paired
# form very slightly overstates a returner's effect, not understates it.


# Whether great players are systematically under-rated. It looks that way and they are not.
#
# Across 2,443 player-seasons with 40 or more matches, comparing each player's real win rate
# with what their Elo implied against the opponents they actually faced, the gap rises with
# the quality of the season: -1.88 percentage points in the weakest fifth, +2.84 in the
# strongest. Jannik Sinner's 2024 season is +4.1, Roger Federer's 2004 +8.4, Rafael Nadal's
# 2005 +10.7. The natural reading is that a rating system cannot keep up with greatness.
#
# It is a lag, not a ceiling. Seasons that improve on the previous one over-perform by +3.47
# points on average; seasons that decline under-perform by -2.42. A rating that trails the
# truth produces exactly that asymmetry, and a genuine ceiling would not. The decisive test
# is persistence: if great players were permanently under-rated, a player who over-performed
# in one season would over-perform in the next. The year-to-year correlation of the gap is
# -0.234 over 1,567 consecutive pairs. It reverses.
#
# So the filter is doing its job, and the residual is the price of not believing a hot start
# too quickly. Federer's peaks are +4.1 in 2005 and +2.7 in 2006, once the rating has caught
# up; his +8.4 is 2004, the year he arrived. The same shape holds for Djokovic in 2011 (+8.4)
# and Nadal in 2005 (+10.7). Sinner's +4.1, +2.2 and +2.3 across 2024 to 2026 is an ordinary
# convergence, not a special case.


# On reacting faster to a genuine step change. The obvious objection to a fixed learning
# rate is that a player who really has improved — a rebuilt serve, a new coach — spends
# thirty matches being under-rated while the filter creeps up on the truth. Two matches of
# spiked hold rate could be a level shift or could be noise, and a fixed gain treats them
# identically.
#
# The way to tell them apart without knowing the event dates is a run: noise is
# independent, so two surprises in the same direction is much stronger evidence of a shift
# than one surprise twice as large. If that holds, the gain should respond to the
# accumulated recent score rather than to each score alone. It does not hold here.
#
# Regressing the next residual on the last one, against on the mean of the last k:
#
#     last match alone     coefficient +0.073, residual variance 1.6983
#     mean of the last 5   coefficient +0.117, residual variance 1.7021
#
# The single last match wins at every k tried. Conditioning on longer runs adds nothing:
# the expected next residual after one above-par match is +0.160, after four in a row it
# is +0.181 — and after four consecutive matches more than three quarters of a standard
# deviation above par it is +0.107, lower than after one. Runs of good matches are mostly
# just runs, and the filter should keep reacting to the newest result and nothing else.
#
# Step changes do happen. Scanning every player with 150 to 350 matches for the single
# best changepoint in hold rate, the median likelihood gain is 4.0, the 90th percentile
# 11.3, and the largest 44.1 (Guillermo Coria). But they are rare enough, and their timing
# uncertain enough, that a gain which chases them costs more on the ordinary matches than
# it recovers on the exceptional ones.


# What actually separates a stable server from a volatile one, and how much of it is
# measurable. Take the obvious pair. On serve points Sinner's standardised residual sd is
# 1.207 and Shelton's 1.400, which looks like a clear difference — Shelton is the streakier
# server. But the sampling error on an estimate from 426 and 228 matches is +/- 0.081 and
# +/- 0.129, and the halves of each career give 1.231 / 1.180 for Sinner against 1.372 /
# 1.401 for Shelton. The intervals overlap. On the raw numbers Shelton looks 16% more
# volatile; the honest reading is that he might be, and a career is not long enough to say.
#
# That is why the estimates are shrunk rather than used raw, and why the stored values come
# out close together: 0.166 for Sinner against 0.175 for Shelton on serve points, 0.122
# against 0.121 on serve percentage. The shrinkage keeps most of a difference that would
# survive replication and discards most of one that would not — 0.5 for the point-winning
# axes, where reliability is 0.36, and 0.75 for serve percentage, where it is 0.79. The
# field-wide spread of the serve-in scale is 0.009 on a mean of 0.133, and that narrowness
# is the finding, not a failure to find one.
#
# Hold rate tells the same story from the other end. Its form spread correlates -0.310 with
# the level, but that is arithmetic, not character: a player holding 90% has less room to
# vary than one holding 70%, and the square-root rule accounts for most of it. Against
# players at their own hold rate, Sinner sits at 8.40pp versus a typical 8.53 and Shelton
# at 8.54 versus 8.20 — both within a fifth of a point of ordinary. The genuinely steady
# ones are elsewhere: Djokovic at 7.17 against 8.44 expected, Alcaraz at 6.75 against 8.69.


# Whether the same dispersion means the same thing for every player. It does not, and
# the model is right not to store it in percentage points.
#
# Break rate ranges from about 10% for a pure server to 31% for an elite returner, and a
# binomial at 10% simply cannot swing as far as one at 30%. Measured across 643 players
# with 150-plus matches, the real form spread in break rate rises with the level, from
# 8.87 percentage points in the bottom fifth to 10.51 in the top — correlation +0.476.
# Jannik Sinner breaks 27.3% of games with a form spread of 8.63pp; Ben Shelton breaks
# 15.1% with a spread of 6.48pp. Same sport, different arithmetic.
#
# So which mapping is right? Two candidate rules predict the observed spread from the
# level, anchored on the bottom band:
#
#     band       observed   sqrt(p(1-p))   p(1-p)
#     20-22%       9.72         9.50        10.17
#     22-24%      10.12         9.81        10.85
#     24-27%      10.40        10.10        11.50
#     27-39%      10.51        10.50        12.44
#
# The square-root rule tracks it; the linear one overshoots badly at the top. A constant
# dispersion on the logit scale implies the linear rule, so storing a single logit scale
# per player would over-swing the best returners. What the data says is constant is the
# *standardised* residual — dispersion measured in units of its own binomial noise — and
# that is what streak.py stores. The logit form spread is flat against level on the return
# side (correlation -0.060), which is the same statement from the other direction.
#
# The practical consequence: Sinner and Shelton get the same dispersion number, 0.174, and
# it is correct that they do. The simulator maps it back through each player's own level,
# so Sinner's return points come out with a 5.6pp spread and Shelton's 5.3pp, without
# either being stored.


# The same tests on the return side. Break rate is not a mirror image of hold rate, and
# the difference is consistent across every measurement:
#
#                            serve      return
#   lag-1 autocorrelation    +0.0745    +0.0747
#   dispersion               1.308      1.314
#   dispersion reliability   0.41       0.26
#   spread across players    4.62pp     4.13pp
#   best-changepoint gain    median 4.0, 90th 11.4    median 4.6, 90th 10.8
#
# The dynamics are identical — same memory, same day-to-day swing, same conclusion that
# runs carry nothing beyond the last match. What differs is how much of it belongs to the
# player. Serving spreads players out more and its variability is far more repeatable,
# because a serve is one player executing alone. Returning is a response to whatever the
# other man hits, so more of its variation is the opponent and less is a stable trait,
# which is exactly why the return dispersion estimate is only 0.26 reliable against 0.41
# for the serve — and why the shrinkage toward the field average is heavier for it.
#
# Ben Shelton is the clean illustration: his hold rate sits at the 75th percentile for
# best changepoint and his break rate at the 15th. The serve moved; the return did not.


# On player-specific learning rates. The tempting next step is to let each player have
# their own a — move the volatile ones faster, the steady ones slower. It does not survive
# testing, and the reason is worth writing down, because it is not the reason it looks
# like from the outside.
#
# The two things are separately identifiable. Dispersion is white noise in the residual:
# a wide day leaves no trace on the next match. A learning rate that is too slow leaves
# autocorrelation, because the filter lags behind a player who is genuinely moving. So
# measure both, and check how repeatable each is within a career:
#
#   dispersion            split-half 0.256, reliability 0.41
#   residual autocorr     split-half 0.060, reliability 0.11
#
# The autocorrelation averages +0.063 with a spread of 0.088 across players, but the
# sampling noise on an estimate from 170 matches is 0.077 — almost the whole spread. And
# out of sample, a per-player rho does WORSE than one pooled value for everyone
# (+0.0082 against +0.0085 in squared error per point). Every player is under-filtered by
# about the same small amount; nobody is under-filtered distinctively.
#
# The one global learning rate is also well pinned down — halving or doubling it costs
# 937 and 528 log-likelihood on the set-strength model — so there is no flat direction
# for a per-player parameter to exploit.
#
# What looks like a fast-moving player is nearly always a wide-dispersion player: the two
# correlate at +0.22, and the highest autocorrelations belong to Becker, Forget, Sampras,
# Ivanisevic and Rusedski. Not players whose level moved unusually fast, but big servers
# whose serve-driven game swings, which is dispersion, already modelled, and carrying no
# memory from one match to the next.


# On the persistence parameter. Gorgi, Koopman and Lit fix it at one — a random walk —
# rather than estimating it, so there is no published value to compare against. Profiling
# the likelihood here says the assumption is right for strength, and wrong for the serve.
#
#   set strength      beta = 1 is a clear peak; 0.999 costs 162 log-likelihood
#   points behind 1st beta = 1 again; 0.999 costs 99
#   first serve in    beta = 0.98 beats a random walk by 1,090, half-life 34 matches
#   second serve in   beta = 0.99 beats it by 289, half-life 69 matches
#
# Which is what you would expect once the two are separated. How good a player is has no
# equilibrium to return to — careers drift, and a random walk is the honest description.
# How often the first serve lands is a physical habit with a level a player comes back to
# after a bad patch, and it mean-reverts over about a season and a half of matches.


def fit_set_strength(matches, until, scalings=(0.0, 0.5, 1.0)):
    best = None
    for scaling in scalings:
        def negll(par):
            try:
                return -filter_set_strength(matches, math.exp(par[0]),
                                            1.0 / (1.0 + math.exp(-par[1])), scaling, until)
            except (OverflowError, ValueError):
                return 1e18
        start = [math.log({0.0: 0.02, 0.5: 0.05, 1.0: 0.08}[scaling]), 6.0]
        res = minimize(negll, start, method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1.0, "maxiter": 60})
        cand = {"negll": res.fun, "a": math.exp(res.x[0]),
                "b": 1.0 / (1.0 + math.exp(-res.x[1])), "scaling": scaling}
        if best is None or cand["negll"] < best["negll"]:
            best = cand
    return best
