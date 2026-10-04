"""Build the model documentation PDF (reproduction spec)."""
import paths as PATHS
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, PageBreak,
                                Preformatted, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus.tableofcontents import TableOfContents

F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", F + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", F + "DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVS", F + "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("DVS-B", F + "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK = colors.HexColor("#1c1a17"); SOFT = colors.HexColor("#6b655a"); ACC = colors.HexColor("#a13d1f"); LINE = colors.HexColor("#d9d2c3")
PANEL = colors.HexColor("#f4efe6")
body = ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=13.6, textColor=INK, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=11.2, textColor=SOFT)
h1 = ParagraphStyle("h1", fontName="DVS-B", fontSize=17, leading=21, textColor=INK, spaceBefore=6, spaceAfter=8, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="DVS-B", fontSize=12.2, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4, keepWithNext=1)
h3 = ParagraphStyle("h3", fontName="DV-B", fontSize=9.6, leading=13, textColor=ACC, spaceBefore=6, spaceAfter=2)
code = ParagraphStyle("code", fontName="DVM", fontSize=6.6, leading=8.9, textColor=INK, backColor=PANEL, borderPadding=(5, 6, 5, 6),
                      spaceBefore=4, spaceAfter=8, leftIndent=4, rightIndent=4)
formula = ParagraphStyle("formula", fontName="DVM", fontSize=8.6, leading=12.5, textColor=INK, leftIndent=14, spaceBefore=2, spaceAfter=6)
cell = ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.4, textColor=INK)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DV-B")
title = ParagraphStyle("title", fontName="DVS-B", fontSize=26, leading=31, textColor=INK, spaceAfter=10)

story = []
def P(t, s=body): story.append(Paragraph(t, s))
def H1(t):
    p = Paragraph(t, h1); p._toc = (0, t); story.append(p)
def H2(t):
    p = Paragraph(t, h2); p._toc = (1, t); story.append(p)
def H3(t): story.append(Paragraph(t, h3))
def EQ(*lines):
    for l in lines: story.append(Paragraph(l.replace(" ", "&nbsp;") if l.startswith("  ") else l, formula))
def CODE(t): story.append(Preformatted(t.strip("\n"), code))
def BUL(items):
    for it in items: story.append(Paragraph(it, ParagraphStyle("b", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2), bulletText="•"))
    story.append(Spacer(1, 3))
def TBL(rows, widths, head=True, align_right=()):
    data = [[Paragraph(str(c), cellb if (head and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    story.append(t); t.setStyle(TableStyle(st)); story.append(Spacer(1, 7))

class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
                         title="Tennis serve/return rating model: technical documentation", author="Wouter van der Harst")
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=self.deco)])
    def deco(self, c, d):
        if d.page == 1: return
        c.saveState(); c.setFont("DV", 7.5); c.setFillColor(SOFT)
        c.drawString(20 * mm, 10 * mm, "Tennis serve/return rating model · technical documentation · 4 October 2026")
        c.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{d.page}"); c.restoreState()
    def afterFlowable(self, f):
        if hasattr(f, "_toc"): self.notify("TOCEntry", (f._toc[0], f._toc[1], self.page))

# ======================================================================================= title
story.append(Spacer(1, 40 * mm))
P("Tennis serve/return rating model", title)
P("Technical documentation and reproduction specification · ATP and WTA", ParagraphStyle("st", parent=body, fontSize=13, leading=17, textColor=SOFT))
story.append(Spacer(1, 8 * mm))
P("Version of 4 October 2026. Data through 29 September 2026 (men) and 30 September 2026 (women).", body)
P("This document describes every data rule, formula, parameter value and processing step of the model, in the order the code "
  "runs them, so that the model can be rebuilt from this document and the raw match files alone. Section 12 gives reference values "
  "a re-implementation should reproduce.", body)
story.append(Spacer(1, 6 * mm))
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9.5, leading=14, leftIndent=0),
                   ParagraphStyle("t1", fontName="DV", fontSize=8.6, leading=12, leftIndent=12, textColor=SOFT)]
story.append(toc)
story.append(PageBreak())

# ======================================================================================= 1 overview
H1("1. Overview")
P("Every player has a <b>serve rating</b> and a <b>return rating</b>, each with a hard, clay and grass version (a surface <i>gap</i> "
  "on top of the general rating). A match is predicted the way it is played: each player's chance of winning a point on serve "
  "comes from their serve rating against the opponent's return rating; points give service holds; holds give sets, with tiebreaks "
  "played point by point; sets give the match. A set-level <b>closing</b> rating, a fixed calibration and a head-to-head term are "
  "applied on the way. After each match the ratings move by how many more serve points and return points the player won than "
  "the model expected (a score-driven, or GAS, update). A thin layer of <b>corrections</b> (age, height on fast courts, court "
  "speed × ace skill, deciding-set clutch, entry type, rest and so on), fitted by logistic regression, sits on top.")
TBL([["Held-out matches", "Logloss", "Accuracy", "Matches"],
     ["Men, this model", "0.60960", "66.10%", "42,218 (27 Feb 2017 – 29 Sep 2026)"],
     ["Men, previous production (v21g)", "0.61120", "65.98%", "same"],
     ["Men, Elo (Tennis Abstract style, calibrated)", "0.63389", "63.44%", "same"],
     ["Women, this model", "0.59097", "68.06%", "19,005 (8 Oct 2018 – 30 Sep 2026)"],
     ["Women, previous production", "0.59504", "67.93%", "same"],
     ["Women, Elo (calibrated)", "0.61572", "65.84%", "same"]], [62, 22, 22, 64])
P("Most of the gain over the previous production model comes from three changes to the rating machinery: (1) predicting through "
  "service points → holds → sets with a <i>shrunk</i> point edge, (2) a faster start for a player's first matches, (3) no rating "
  "decay between matches; and (4) learning from the points played in retired matches, for the player who did not retire. Splitting into serve and return is neutral for win odds (a single rating with the same machinery scores "
  "within 0.0001) but makes ratings interpretable; the surface-gap structure guarantees that a player's general rating always "
  "lies between their worst and best surface rating.")

# ======================================================================================= 2 notation
H1("2. Notation and conventions")
BUL(["σ(x) = 1 / (1 + e<super>−x</super>) is the logistic function; logit(p) = ln(p / (1 − p)).",
     "Ratings are stored in <b>game-logit units</b> (the scale of the original single-rating model, in which σ(rating difference) "
     "was the chance to win a game). The point scale is obtained by dividing by <b>PT_M = 2.132</b>.",
     "Data rows are written from the winner's side: <b>a</b> = winner, <b>b</b> = loser. The model itself is symmetric; the label is "
     "only used to read the outcome. Every probability below is P(a wins), and every correction feature is (value for a) − (value for b).",
     "clip(x, lo, hi) = min(max(x, lo), hi). All logs are natural logs.",
     "Dates are calendar dates; 'days' are whole days between tournament start dates (the files carry only the tournament date)."])

# ======================================================================================= 3 data
H1("3. Data")
H2("3.1 Sources")
TBL([["", "Men (ATP)", "Women (WTA)"],
     ["Files", "TennisMyLife ATP database, one CSV per year 1968–2023 (tour level: Grand Slams, Masters, 500, 250, finals, Davis Cup, Olympics), plus uploaded 2024, 2025 and 2026 tour files and 2024–2026 challenger files (2026 to 29 Sep 2026). Challenger matches therefore enter only from 2024. A file named ATP_Database.csv and any file whose name contains 'qualifying_matches' are skipped.",
      "TennisMyLife WTA main-tour files, one per year 1990–2026, plus the 'ongoing tourneys' file (to 30 Sep 2026). No ITF, no separate qualifying files."],
     ["Format", "Jeff Sackmann column layout: tourney_id, tourney_name, surface, tourney_level, indoor, tourney_date (YYYYMMDD), winner_/loser_ name, seed, entry, hand, ht, ioc, age, rank; score, best_of, round, minutes; w_/l_ ace, df, svpt, 1stIn, 1stWon, 2ndWon, SvGms, bpSaved, bpFaced.", "Same layout."],
     ["Rated matches", "211,014", "94,973"],
     ["Birth dates", "Per player name, one birth date (ordinal day). Where unknown: tournament date − age × 365.25 from the age column.", "Same rule (file wta_birth_ordinal_v2)."]],
    [22, 74, 74])
H2("3.2 Row construction")
BUL(["<b>Date</b>: tourney_date parsed as YYYYMMDD; rows without a valid date are dropped.",
     "<b>Players</b>: winner_name and loser_name, stripped; rows with a missing name or the same name twice are dropped.",
     "<b>Duplicates</b>: rows with the same (date, winner, loser, score) are kept once (first occurrence).",
     "<b>Level</b>: tourney_level C or S → challenger; Q → qualifying; anything else → tour. The raw code is kept as <i>lvl</i> "
     "(G = Grand Slam, M = Masters/1000, F = finals, A/250/500 = other). WTA coding: T1, PM, P5 and 1000 → M; F, YEC, or W with "
     "'Championships'/'Finals' in the name → F; Q1–Q4 rounds → Q; G stays G; 'I' (2009–2020) means International, a small event.",
     "<b>Surface</b>: Hard, Clay, Grass or Carpet; anything else → none. Carpet and none have no surface rating (general only); "
     "their surface-average key is 'Carpet' and 'Hard' respectively.",
     "<b>Indoor</b>: indoor column equals 'I'. <b>Best of</b>: best_of (default 3).",
     "<b>Ordering</b>: rows sorted by (date, tourney_id, qualifying before main draw, round order) with round order Q1 −3, Q2 −2, "
     "Q3 −1, Q4 −1, RR 0, R128 0, R64 1, R32 2, R16 3, QF 4, SF 5, BR 6, F 6 (unknown 2). The WTA history export uses the same order."])
H2("3.3 Which matches are rated")
P("A row is <b>rated</b> when (i) its score contains none of RET, W/O, DEF, ABD, UNFINISHED; (ii) the set tokens (regex "
  "<font face='DVM'>^(\\d+)-(\\d+)</font> on each space-separated token) give at least two sets; (iii) the winner won more sets "
  "than the loser. Sets won (w, l) count tokens with x &gt; y and y &gt; x; games won (gw, gl) sum x and y over the tokens. "
  "Unrated rows still feed rest days, injury flags, event fatigue and the profile pages.")
P("<b>Retirements and walkovers</b> (score contains RET or W/O, no DEF/ABD) enter the rating replay as <b>update-only rows</b> at "
  "their place in the order of 3.2 (section 5.12). They are never scored, never counted in head-to-heads and never change the closing "
  "rating. A retirement with valid serve points (3.4) updates only the player who did <i>not</i> retire; a walkover, or a retirement "
  "without stats, updates no rating but counts as a match for layoff dates, age drift and the match counter.")
H2("3.4 Cleaning rules")
BUL(["<b>Swapped serve stats</b>. A row's stats are attached to the wrong player when the winner took more than 55% of the "
     "games but less than 47% of the points, where points won by the winner = w_1stWon + w_2ndWon + (l_svpt − l_1stWon − l_2ndWon). "
     "Such rows (ATP 2019: 30, 2025: 45; WTA 2021: 33, 2025: 43; none in clean years) get every w_* and l_* column swapped before use.",
     "<b>Reversed WTA scores</b>. 78 WTA rows had the score written from the loser's side (sets won by the listed winner fewer than "
     "by the loser). Each set token is flipped (x-y → y-x, tiebreak suffix kept) and all w_/l_ stats of those rows are blanked.",
     "<b>Serve points</b>. For a rated row, srv_won(a) = w_1stWon + w_2ndWon, srv_n(a) = w_svpt, ret_won(a) = l_svpt − l_1stWon − l_2ndWon, "
     "ret_n(a) = l_svpt. Points are treated as missing when total points w_svpt + l_svpt &lt; 40, or when 1stWon + 2ndWon exceeds svpt for either side.",
     "<b>Height</b> counts only between 150 and 220 cm; otherwise missing (replaced by the tour mean height HT_MEAN, computed over all rows)."])

# ======================================================================================= 4 state
H1("4. Player state")
TBL([["Symbol", "Meaning", "Start value"],
     ["S<sub>p</sub>, R<sub>p</sub>", "general serve and return rating (game-logit units)", "debut value v (4.1)"],
     ["g<sup>S</sup><sub>p,x</sub>, g<sup>R</sup><sub>p,x</sub>", "surface gap on surface x ∈ {Hard, Clay, Grass} for serve and return", "0"],
     ["c<sub>p</sub>", "closing (set-level) rating", "0"],
     ["n<sub>p</sub>", "number of rated matches played so far", "0"],
     ["last<sub>p</sub>", "date of the player's previous rated match", "none"],
     ["μ<sub>x</sub>", "surface serve average (logit of the average serve-point win rate), per surface key incl. Carpet", "0.55"],
     ["H<sub>p,q</sub>", "head-to-head wins of p over q in rated matches", "0"]], [32, 100, 38])
P("The player's <b>level</b> (the single number shown as 'rating') is (S + R) / 2. The surface skill on x is S + λ·g<sup>S</sup><sub>x</sub> "
  "for serve and R + λ·g<sup>R</sup><sub>x</sub> for return; the gaps are kept so that 0.6·g<sub>Hard</sub> + 0.3·g<sub>Clay</sub> + 0.1·g<sub>Grass</sub> = 0, "
  "which makes the general level exactly the 60/30/10 weighted average of the three surface levels.")
H2("4.1 Debut value")
P("When a player appears in a rated match for the first time:")
EQ("v = clip(0.05 − 0.15·(ln rank − ln 100), −1.5, 1.0)     if the row gives the player's official rank",
   "v = −0.05 (tour level), −0.20 (challenger or qualifying)  otherwise",
   "S = R = v ;  all surface gaps = 0 ;  c = 0")

# ======================================================================================= 5 per-match
H1("5. The per-match algorithm")
P("Rated rows are processed one at a time in the order of 3.2. For row (a, b, date, surface x, best-of k) the steps below run "
  "in exactly this order. Everything a prediction uses is fixed before the row's outcome is read.")
H2("5.1 Layoff decay")
P("For each player with a previous rated match: idle = max(0, days since last − 30). If idle &gt; 0, multiply S, R, every "
  "surface gap and c by 0.9995<super>idle</super>. (Half-life about 1,386 days beyond the 30-day grace.)")
H2("5.2 Age drift")
P("For each player with a previous rated match and a known age A (years, = (date − birth) / 365.25): let y = min(days since last / 365.25, 1).")
EQ("drift = +0.05·y              if A < 24",
   "drift = −0.01·(A − 30)·y     if A > 30",
   "drift = 0                    otherwise",
   "S ← clip(S + drift, −3, 3) ;  R ← clip(R + drift, −3, 3)     (gaps unchanged)")
P("Then last<sub>a</sub> = last<sub>b</sub> = date.")
H2("5.3 Effective serve and return ratings")
P("With λ = 0.6 (men) or 0.7 (women) and surface x ∈ {Hard, Clay, Grass}:")
EQ("s_a = S_a + λ·gS_a,x    r_a = R_a + λ·gR_a,x      (same for b)",
   "for Carpet or no surface:  s = S, r = R")
H2("5.4 Point probabilities")
P("μ = μ<sub>x</sub> (key 'Hard' when the surface is missing). The <b>update</b> probabilities use the full point edge; the "
  "<b>prediction</b> probabilities shrink it by h (h = 0.7 men, 0.6 women):")
EQ("p_a = clip(σ(μ + (s_a − r_b)/2.132), 0.02, 0.98)      a's chance per point on serve (update)",
   "p_b = clip(σ(μ + (s_b − r_a)/2.132), 0.02, 0.98)",
   "h_a = clip(σ(μ + h·(s_a − r_b)/2.132), 0.02, 0.98)   (prediction)",
   "h_b = clip(σ(μ + h·(s_b − r_a)/2.132), 0.02, 0.98)")
H2("5.5 Hold, set and match probabilities")
P("Chance to hold serve with point probability p (q = 1 − p), deuce included:")
EQ("hold(p) = p<super>4</super>·(1 + 4q + 10q<super>2</super>) + 20·p<super>3</super>q<super>3</super> · p<super>2</super> / (1 − 2pq)")
P("Set probability SET(p<sub>A</sub>, p<sub>B</sub>) for A serving first, by recursion over games (x, y) with A serving when x + y is even:")
EQ("G(x,y) = 1 if x ≥ 6 and x − y ≥ 2 ;  0 if y ≥ 6 and y − x ≥ 2 ;  1/0 if x = 7 or y = 7 (whoever has 7)",
   "G(6,6) = TB(0,0)",
   "G(x,y) = P·G(x+1,y) + (1−P)·G(x,y+1),  P = hold(p_A) if x+y even, else 1 − hold(p_B)")
P("Tiebreak TB over points (x, y), A serving when (x + y) mod 4 ∈ {0, 3}:")
EQ("TB(x,y) = 1 if x ≥ 7 and x − y ≥ 2 ;  0 if y ≥ 7 and y − x ≥ 2",
   "TB(x,x) for x ≥ 6 = w / (w + l),  w = p_A·(1 − p_B), l = (1 − p_A)·p_B",
   "TB(x,y) = P·TB(x+1,y) + (1−P)·TB(x,y+1),  P = p_A if A serves, else 1 − p_B")
P("The set probability used averages over who serves first; the point probabilities are rounded to 3 decimals first (exactly "
  "as implemented, for caching):")
EQ("q_set = clip( ½·[SET(r3(h_a), r3(h_b)) + 1 − SET(r3(h_b), r3(h_a))], 10<super>−6</super>, 1 − 10<super>−6</super>)")
P("Match from set probability q (k = best of 3 needs n = 2 sets, best of 5 needs n = 3):")
EQ("MATCH(q, k) = Σ<sub>l=0..n−1</sub> C(n+l−1, l) · q<super>n</super> · (1−q)<super>l</super>")
H2("5.6 Closing, calibration, head-to-head")
EQ("q_adj   = clip(σ(logit(q_set) + c_a − c_b), 10^−6, 1 − 10^−6)",
   "q_match = clip(MATCH(q_adj, k), 10^−6, 1 − 10^−6)",
   "z_model = CAL_I + CAL_S · logit(q_match),   CAL_I = 0.010096,  CAL_S = 0.915525",
   "if a and b met before in rated matches (wins w_a, w_b; n = w_a + w_b):",
   "   z_model += slope_h2h(n) · logit((w_a + 0.5) / (n + 1))",
   "   slope_h2h: n = 1 → 0.022474 ;  n = 2 → −0.002023 ;  n = 3–4 → 0.039395 ;  n ≥ 5 → 0.053349")
P("z_model is the rating model's log-odds that a wins (before the corrections layer). Then H<sub>a,b</sub> += 1.")
H2("5.7 Closing update")
P("The closing rating learns from sets won against a calibrated set expectation <i>without</i> closing:")
EQ("q_mr = clip(MATCH(q_set, k), 10^−6, 1 − 10^−6)",
   "q_mc = clip(σ(CAL_I + CAL_S·logit(q_mr)), 10^−6, 1 − 10^−6)",
   "q_sc = INV_MATCH(round(q_mc, 4), k)      (bisection on q ∈ [0.001, 0.999], 40 halvings, MATCH(q,k) = q_mc)",
   "e_c  = sets_a − (sets_a + sets_b) · q_sc",
   "c_a ← 0.999·c_a + 0.005·e_c ;   c_b ← 0.999·c_b − 0.005·e_c")
H2("5.8 Surprises")
P("Games surprise (always computed), with the update hold probabilities:")
EQ("q_g = clip(½·[hold(p_a) + 1 − hold(p_b)], 10^−6, 1 − 10^−6)",
   "e_g = gw − (gw + gl)·q_g ;   u_S = u_R = ½·e_g")
P("If the row has serve points (srv_won, srv_n, ret_won, ret_n for a):")
EQ("e_S = (srv_won − srv_n·p_a) · 0.7 / 2.132          a's serve points above expectation",
   "e_R = (ret_won − ret_n·(1 − p_b)) · 0.7 / 2.132    a's return points above expectation",
   "u_S = (1 − w_pts)·½·e_g + w_pts·e_S ;   u_R = (1 − w_pts)·½·e_g + w_pts·e_R",
   "w_pts = 1.0 (men),  0.75 (women)")
P("Surface average (only rows with points; uses μ before this row's update):")
EQ("μ_x ← μ_x + 0.0005 · [ (srv_won + ret_n − ret_won) / (srv_n + ret_n) − σ(μ_x) ]")
H2("5.9 Step size")
EQ("step = 0.0089679476 · κ ,   κ = 0.8 (men), 1.0 (women)",
   "m_p  = 1.4 if age < 24 (else 1.0)   ×   [1 + 2·exp(−n_p / 15)]",
   "n_p ← n_p + 1  (after m_p is computed)")
P("The second factor is the early-career gain: triple speed on a player's first match, settling over about 15 matches.")
H2("5.10 General rating update (no decay)")
P("A's serve surplus is b's return deficit and vice versa; each half gets twice the half-surprise so that the level "
  "(S + R)/2 moves exactly like a single rating:")
EQ("S_a ← S_a + step·m_a·2u_S        R_a ← R_a + step·m_a·2u_R",
   "S_b ← S_b − step·m_b·2u_R        R_b ← R_b − step·m_b·2u_S")
H2("5.11 Surface gap update and re-centring")
P("Only when x ∈ {Hard, Clay, Grass}; γ = gap step = 0.6 (men), 0.4 (women). For player a (b analogous with its own deltas "
  "−2u<sub>R</sub> for serve and −2u<sub>S</sub> for return):")
EQ("gS_a,x ← gS_a,x + step·m_a·γ·2u_S ;   gR_a,x ← gR_a,x + step·m_a·γ·2u_R",
   "then for each of gS and gR separately:",
   "   ḡ = 0.6·g_Hard + 0.3·g_Clay + 0.1·g_Grass ;   g_y ← g_y − ḡ  for y ∈ {Hard, Clay, Grass}")
P("The surface match counters (used only for display) increase by one for both players.")
H2("5.12 Update-only rows (retirements and walkovers)")
P("Steps 5.1–5.5 run as for a rated row (layoff decay, age drift, last date, predictions). Then: no head-to-head count, no closing "
  "update (5.7), no games surprise. If the row has serve points: u<sub>S</sub> = e<sub>S</sub> and u<sub>R</sub> = e<sub>R</sub> (pure points, any tour), "
  "μ is updated as in 5.8, n<sub>a</sub> and n<sub>b</sub> increase, and steps 5.10–5.11 run with m<sub>b</sub> = 0, so only the winner "
  "(the player who did not retire) moves. Without serve points (walkovers, retirements without stats): m<sub>a</sub> = m<sub>b</sub> = 0, so no rating moves, "
  "but n<sub>a</sub>, n<sub>b</sub> and the last dates are updated.")

# ======================================================================================= 6 corrections
H1("6. Corrections layer")
P("The final log-odds that a wins is a logistic regression on z<sub>model</sub> plus 13 features, each the difference "
  "(a's value − b's value), all computed from information available before the match:")
EQ("z = w_0·z_model + Σ_k w_k·Δx_k")
H2("6.1 Features")
TBL([["Feature", "Per-player value"],
     ["rank_missing", "1 if the row has no official rank for the player"],
     ["rest_log", "ln(1 + min(rest, 60)); rest = days from the player's previous match of any kind (rated or not) to this event, measured at the player's first match of the event; 21 if no previous match"],
     ["home", "1 if the player's IOC country equals the event's host country; host = most common IOC among the event's wildcard (WC) entrants, needing at least two"],
     ["q_entry", "1 if entry = Q (qualifier)"],
     ["big_x_age27", "max(0, age − 27) × [lvl ∈ {G, M}]"],
     ["height_x_fast", "(height − HT_MEAN)/10 × fast, fast = 1 on grass or indoors"],
     ["fat_sets_x_age27", "sets already played by the player earlier in this event × max(0, age − 27)"],
     ["age_over32", "max(0, age − 32); men: capped at 3 (i.e. min(max(0, age − 32), 3))"],
     ["prev_ret_loss", "1 if the player's previous match ended in a retirement or walkover loss"],
     ["layoff90", "1 if rest ≥ 90 days"],
     ["speedw_x_ace_gap", "court speed deviation × ace skill (6.2); one feature for the pair: (speed − centre) × (ln ace_a − ln ace_b)"],
     ["deciding_set_clutch", "clutch (6.3)"],
     ["slam_x_atp_top4", "[lvl = G] × [official rank ≤ 4]"]], [36, 134])
P("Age defaults to 26 when unknown; height defaults to HT_MEAN.", small)
H2("6.2 Court speed and ace skill")
P("Built in one pass over all rows (rated or not) in date order. A row with ace and serve-point counts for both players (both "
  "serve-point totals &gt; 10) updates the state; every rated row first reads the pre-match estimate.")
BUL(["Court type = 'Indoor ' + surface if indoor, else surface ('Hard' when missing). Event key = (level, tourney_id without its year prefix, court type).",
     "League ace rate L (start 0.07); player ace skill A<sub>p</sub> and ace concession C<sub>p</sub> (multiplicative, start 1).",
     "Expected aces: E = L·A<sub>a</sub>·C<sub>b</sub>·svpt<sub>a</sub> + L·A<sub>b</sub>·C<sub>a</sub>·svpt<sub>b</sub>; actual X = ace<sub>a</sub> + ace<sub>b</sub>. If E &gt; 0.5: "
     "match log ratio lr = ln((X + 0.5)/(E + 0.5)), weight W = svpt<sub>a</sub> + svpt<sub>b</sub>; append (year, lr, W) to the event's history; "
     "surface running mean: m<sub>c</sub> ← 0.999·m<sub>c</sub> + 0.001·lr·W, w<sub>c</sub> ← 0.999·w<sub>c</sub> + 0.001·W.",
     "Skill update per server s vs returner t with court factor f = exp(clip(lr, −1, 1)): ratio = (aces<sub>s</sub> + 0.5)/(L·A<sub>s</sub>·C<sub>t</sub>·f·svpt<sub>s</sub> + 0.5); "
     "A<sub>s</sub> ← A<sub>s</sub>·ratio<super>0.03</super>; C<sub>t</sub> ← C<sub>t</sub>·ratio<super>0.015</super>. Then L ← 0.9995·L + 0.0005·X/W.",
     "Pre-match speed of an event in year Y: (1500·m<sub>c</sub>/w<sub>c</sub> + Σ lr<sub>i</sub>·W<sub>i</sub>·0.6<super>Y−y<sub>i</sub></super>) / (1500 + Σ W<sub>i</sub>·0.6<super>Y−y<sub>i</sub></super>) over the event's past editions; "
     "'has speed' only when the event has history.",
     "Centre = mean speed of rated training rows of the same court type that have speed. Feature = (speed − centre)·(ln A<sub>a</sub> − ln A<sub>b</sub>) when the event has speed history, else 0."])
H2("6.3 Deciding-set clutch")
P("Running over rated rows in order, with z<sub>model</sub> from 5.6:")
EQ("feature = D_a/(N_a + 50) − D_b/(N_b + 50)      (read before the update)",
   "if the match went to a deciding set (3 set tokens in best of 3, 5 in best of 5):",
   "   q = INV_MATCH(round(clip(σ(z_model), 0.001, 0.999), 4), k)",
   "   D_a += 1 − q ;  D_b −= 1 − q ;  N_a += 1 ;  N_b += 1")
H2("6.4 Fitting")
P("Weights w = (w<sub>0</sub>, w<sub>1..13</sub>) minimise, over the training rows (all rated rows before the test start, 8.1), "
  "with the target 'a wins' on every row:")
EQ("Σ ln(1 + e^(−z)) + 5·10^−5·Σ_{k≥1} w_k<super>2</super>      (no intercept; w_0 unpenalised; L-BFGS-B from w_0 = 1, others 0)")
P("(The gradient used adds 10<super>−4</super>·w<sub>k</sub>, i.e. exactly the derivative of the penalty.) Fitted values:")
TBL([["Weight", "Men", "Women"],
     ["w_0 (on z_model)", "0.9865", "1.0983"],
     ["rank_missing", "−0.993", "−1.214"], ["rest_log", "−0.112", "−0.121"], ["home", "0.215", "0.239"], ["q_entry", "0.344", "0.400"],
     ["big_x_age27", "−0.0235", "−0.0304"], ["height_x_fast", "0.0676", "−0.0030"], ["fat_sets_x_age27", "−0.0037", "0.0005"],
     ["age_over32", "0.0162", "−0.0032"], ["prev_ret_loss", "−0.223", "−0.310"], ["layoff90", "0.0091", "−0.0415"],
     ["speedw_x_ace_gap", "0.333", "0.260"], ["deciding_set_clutch", "0.582", "0.230"], ["slam_x_atp_top4", "0.200", "0.191"]], [60, 30, 30])

# ======================================================================================= 7 final
H1("7. Final prediction")
EQ("P(a beats b | match completed) = σ(z),   z = w_0·z_model + Σ_k w_k·Δx_k",
   "P(a advances) = (1 − r)·σ(z) + r·σ(β·z)")
P("r and β are the retirement/walkover risk per match and how much strength still matters when that happens, estimated on the "
  "training years (r = share of tour-level matches ending in RET or W/O; β = logistic slope of who advanced in those matches on z):")
TBL([["", "r (Slams)", "β (Slams)", "r (other)", "β (other)"],
     ["Men", "3.27%", "0.377", "2.68%", "0.110"],
     ["Women", "1.30%", "0.530", "3.04%", "0 (fitted −0.047, floored at 0)"]], [30, 30, 30, 30, 50])
P("In matches that end early, the player who goes through led at the moment of retirement 85% (men) and 86% (women) of the time, "
  "but before the match strength barely predicts it (better-ranked player advances 51–53%, against 67% in completed matches). "
  "The odds page shows P(a advances) as the headline and P(match completed) in the breakdown.")
P("For the live odds and ratings page the state is taken at the last date in the data, with layoff decay applied up to that "
  "date (5.1) but no age drift. Match-context features without a scheduled match (rest, home, entry, injury, fatigue, layoff) "
  "are set to neutral (zero difference); the user picks surface, indoor, level (tour event, 1000, Grand Slam), best-of and court speed.")

# ======================================================================================= 8 evaluation
H1("8. Evaluation")
H2("8.1 Protocol")
TBL([["Split", "Men", "Women"],
     ["Training-fit", "143,420 matches, 1968 – 21 Apr 2008", "64,567 matches, 1990 – 14 Jul 2014"],
     ["Selection years", "25,376 matches, 28 Apr 2008 – 20 Feb 2017", "11,401 matches, 21 Jul 2014 – 1 Oct 2018"],
     ["Held-out test", "42,218 matches, 27 Feb 2017 – 29 Sep 2026", "19,005 matches, 8 Oct 2018 – 30 Sep 2026"]], [34, 68, 68])
P("Test start = the date at the 80% quantile of rated rows; selection = the last 15% of the remaining (training) rows. Ratings are "
  "always causal (each prediction uses only earlier matches). Every choice of structure and parameter is made on the "
  "<b>selection years</b> (corrections fitted on training-fit rows only, scored on selection rows); the test is scored once, "
  "with corrections refitted on all training rows. A third number, 'whole dataset 2005+', fits the corrections on all rows from "
  "2005 and scores the same rows; it is a stability check, not a test. Metric: mean logloss of the winner (lower is better); accuracy = share of matches where the model's favourite won.")
H2("8.2 Development path")
TBL([["Step", "Men sel. / test", "Women sel. / test"],
     ["Previous production rating, corrections refitted", "0.55386 / 0.61209", "0.60457 / 0.59504"],
     ["Same machinery rebuilt from scratch (exact reproduction)", "0.55386 / 0.61209", "0.60522 / 0.59576 *"],
     ["+ early-career gain", "0.55341 / 0.61127", "0.60422 / 0.59443"],
     ["+ no decay", "0.55303 / 0.61095", "0.60411 / 0.59427"],
     ["+ serve/return split, hold-based prediction, h = 0.7 / 0.6", "0.55256 / 0.61068", "0.60246 / 0.59298"],
     ["+ triple early gain (final blend model)", "0.55241 / 0.61039", "0.60221 / 0.59239"],
     ["Surface = general + gap (λ, γ = 0.6, 0.6 / 0.7, 0.4)", "0.55258 / 0.61036", "0.60300 / 0.59190"],
     ["+ retirements/walkovers as dated rows (no update)", "0.55254 / 0.61035", "0.60314 / 0.59191"],
     ["+ partial points of retirements, both players", "0.55231 / 0.61025", "0.60306 / 0.59162"],
     ["Final: partial points, non-retiring player only", "0.55144 / 0.60959", "0.60275 / 0.59097"]], [86, 42, 42])
P("* The women's previous production also had a separate serve/return add-on; the rebuild is of the main rating only.", small)
H2("8.3 What did not help (tested, rejected)")
BUL(["Cross-interaction between players' serve and return ratings (Koopman–Lit style) and a slower-learning serve/return 'style': flat.",
     "Faster surface updates (×1.25 to ×2), or a boost for a player's first matches on a surface: worse on both tours.",
     "Separate serve and return blend weights, venue-specific serve averages, separate serve and return steps: at most 0.0005.",
     "Level-shift repairs when the general rating left the surface range: equal to the gap model (which was chosen for clarity).",
     "Elo, rolling Elo, record vs top-20, volatility, elite holders, break-point conversion, Slam weighting, serve speed proxies, "
     "a returner-specific sensitivity to ace servers (reliability 0.34 men, 0.13 women; gain ≤ 0.0003): none survived the selection years.",
     "Betting (tested with an earlier model version on 2022–2026 tennis-data.co.uk odds): Pinnacle closing odds are sharper (logloss 0.585 vs the model's 0.598 on the same matches) and no betting strategy beat the market after removing the bookmaker margin."])

# ======================================================================================= 9 parameters
H1("9. Parameter reference")
TBL([["Parameter", "Men", "Women", "Where"],
     ["Point scale PT_M", "2.132", "2.132", "5.4, 5.8"],
     ["Points weight in surprise PT_C", "0.7", "0.7", "5.8"],
     ["Points share w_pts", "1.0", "0.75", "5.8"],
     ["Base step", "0.0089679476", "0.0089679476", "5.9"],
     ["Step scale κ", "0.8", "1.0", "5.9"],
     ["Young boost (age < 24)", "1.4", "1.4", "5.9"],
     ["Early gain (first match) / settle", "3 / 15 matches", "3 / 15 matches", "5.9"],
     ["Persistence (decay per match)", "1 (none)", "1 (none)", "5.10"],
     ["Prediction shrink h", "0.7", "0.6", "5.4"],
     ["Surface gap weight λ", "0.6", "0.7", "5.3"],
     ["Surface gap step γ", "0.6", "0.4", "5.11"],
     ["Gap centring weights", "0.6 / 0.3 / 0.1", "0.6 / 0.3 / 0.1", "5.11"],
     ["Point clip", "[0.02, 0.98]", "[0.02, 0.98]", "5.4"],
     ["μ start / step", "0.55 / 0.0005", "0.55 / 0.0005", "5.8"],
     ["Layoff grace / factor", "30 days / 0.9995", "30 days / 0.9995", "5.1"],
     ["Age drift young / old", "+0.05/yr < 24 ; −0.01·(A−30)/yr > 30", "same", "5.2"],
     ["Rating clip (drift)", "[−3, 3]", "[−3, 3]", "5.2"],
     ["Debut from rank", "0.05 − 0.15·(ln rank − ln 100), clip [−1.5, 1]", "same", "4.1"],
     ["Debut without rank", "tour −0.05, chall/qual −0.20", "same", "4.1"],
     ["Closing a / b", "0.005 / 0.999", "0.005 / 0.999", "5.7"],
     ["Calibration CAL_I / CAL_S", "0.010096 / 0.915525", "same", "5.6"],
     ["H2H slopes n=1, 2, 3–4, 5+", "0.02247, −0.00202, 0.03939, 0.05335", "same", "5.6"],
     ["Clutch shrink", "50 pseudo-matches", "same", "6.3"],
     ["Corrections ridge", "5·10^−5", "5·10^−5", "6.4"],
     ["Retired-match points weight / side", "1 / non-retiring player", "1 / non-retiring player", "5.12"],
     ["Retirement risk r, slope β (Slam; other)", "3.27%, 0.377; 2.68%, 0.110", "1.30%, 0.530; 3.04%, 0", "7"]], [52, 50, 40, 28])

# ======================================================================================= 10 pseudocode
H1("10. Complete pseudocode")
P("The rating engine for one tour, written out end to end (the corrections layer and the exports follow sections 6–7).")
CODE(r'''
def replay(rows, P):            # rows: rated rows in order; P: tour parameters (section 9)
    S, R = {}, {}; gS = {x: {} for x in SURF}; gR = {x: {} for x in SURF}
    c, n, last, H = {}, {}, {}, defaultdict(int); mu = defaultdict(lambda: 0.55)
    out = []
    for r in rows:                      # rated rows AND update-only rows (retirements, walkovers), in order
        a, b, x, k = r.winner, r.loser, r.surface, r.best_of
        for p in (a, b):                                   # 4.1 debut
            if p not in S:
                v = debut(r.rank[p], r.level)
                S[p] = R[p] = v; c[p] = 0; n[p] = 0
                for y in SURF: gS[y][p] = gR[y][p] = 0
        for p in (a, b):                                   # 5.1 layoff decay
            if p in last:
                idle = max(0, (r.date - last[p]).days - 30)
                if idle:
                    f = 0.9995 ** idle; S[p] *= f; R[p] *= f; c[p] *= f
                    for y in SURF: gS[y][p] *= f; gR[y][p] *= f
        for p in (a, b):                                   # 5.2 age drift
            if p in last and age(p, r.date) is not None:
                yrs = min((r.date - last[p]).days / 365.25, 1); A = age(p, r.date)
                d = 0.05 * yrs if A < 24 else (-0.01 * (A - 30) * yrs if A > 30 else 0)
                S[p] = clip(S[p] + d, -3, 3); R[p] = clip(R[p] + d, -3, 3)
        last[a] = last[b] = r.date
        def eff(p):                                        # 5.3
            if x in SURF: return S[p] + P.lam * gS[x][p], R[p] + P.lam * gR[x][p]
            return S[p], R[p]
        sa, ra = eff(a); sb, rb = eff(b); m = mu[x or "Hard"]
        pa = clip(sig(m + (sa - rb) / 2.132), .02, .98); pb = clip(sig(m + (sb - ra) / 2.132), .02, .98)
        ha = clip(sig(m + P.h * (sa - rb) / 2.132), .02, .98); hb = clip(sig(m + P.h * (sb - ra) / 2.132), .02, .98)
        qs = clip(0.5 * (SET(r3(ha), r3(hb)) + 1 - SET(r3(hb), r3(ha))), 1e-6, 1 - 1e-6)     # 5.5
        qadj = clip(sig(logit(qs) + c[a] - c[b]), 1e-6, 1 - 1e-6)                          # 5.6
        z = CAL_I + CAL_S * logit(clip(MATCH(qadj, k), 1e-6, 1 - 1e-6))
        wa, wb = H[a, b], H[b, a]
        if wa + wb: z += h2h_slope(wa + wb) * logit((wa + 0.5) / (wa + wb + 1))
        if r.update_only:                                                                  # 5.12
            if not r.points: n[a] += 1; n[b] += 1; continue
            sw, sn, rw, rn = r.points
            uS = (sw - sn * pa) * 0.7 / 2.132; uR = (rw - rn * (1 - pb)) * 0.7 / 2.132
            mu[x or "Hard"] += 0.0005 * ((sw + rn - rw) / (sn + rn) - sig(m))
            ma = mult(a); n[a] += 1; n[b] += 1; step = 0.0089679476 * P.kappa
            S[a] += step * ma * 2 * uS; R[a] += step * ma * 2 * uR          # only the non-retiring player
            if x in SURF: update_and_recentre_gaps(a, ma, 2 * uS, 2 * uR)   # as 5.11
            continue
        out.append(z); H[a, b] += 1
        qmc = clip(sig(CAL_I + CAL_S * logit(clip(MATCH(qs, k), 1e-6, 1 - 1e-6))), 1e-6, 1 - 1e-6)   # 5.7
        e_c = r.sets_w - (r.sets_w + r.sets_l) * INV_MATCH(round(qmc, 4), k)
        c[a] = 0.999 * c[a] + 0.005 * e_c; c[b] = 0.999 * c[b] - 0.005 * e_c
        qg = clip(0.5 * (hold(pa) + 1 - hold(pb)), 1e-6, 1 - 1e-6)                          # 5.8
        uS = uR = 0.5 * (r.games_w - (r.games_w + r.games_l) * qg)
        if r.points:
            sw, sn, rw, rn = r.points
            eS = (sw - sn * pa) * 0.7 / 2.132; eR = (rw - rn * (1 - pb)) * 0.7 / 2.132
            mu[x or "Hard"] += 0.0005 * ((sw + rn - rw) / (sn + rn) - sig(m))
            uS = (1 - P.w_pts) * uS + P.w_pts * eS; uR = (1 - P.w_pts) * uR + P.w_pts * eR
        step = 0.0089679476 * P.kappa                                                       # 5.9
        def mult(p):
            mm = 1.4 if (age(p, r.date) or 99) < 24 else 1.0
            return mm * (1 + 2 * exp(-n[p] / 15))
        ma, mb = mult(a), mult(b); n[a] += 1; n[b] += 1
        dA = (2 * uS, 2 * uR); dB = (-2 * uR, -2 * uS)                                       # 5.10
        S[a] += step * ma * dA[0]; R[a] += step * ma * dA[1]
        S[b] += step * mb * dB[0]; R[b] += step * mb * dB[1]
        if x in SURF:                                                                        # 5.11
            for p, mp, (ds, dr) in ((a, ma, dA), (b, mb, dB)):
                gS[x][p] += step * mp * P.gamma * ds; gR[x][p] += step * mp * P.gamma * dr
                for G in (gS, gR):
                    gbar = 0.6 * G["Hard"][p] + 0.3 * G["Clay"][p] + 0.1 * G["Grass"][p]
                    for y in SURF: G[y][p] -= gbar
    return out
''')
P("Helper definitions: sig = σ; r3 = round to 3 decimals; SET, hold, MATCH as in 5.5; INV_MATCH by bisection as in 5.7; "
  "h2h_slope as in 5.6; debut as in 4.1; age(p, date) = (date − birth)/365.25 or None; mult(p) as defined inside the loop "
  "(define it before first use in a real implementation); update_and_recentre_gaps = the body of 5.11 for one player.", small)

# ======================================================================================= 11 page
H1("11. The ratings page")
BUL(["<b>History charts</b> (men, women): per player and month (weekly for about 170 selected players) the pre-match state: level, level "
     "on the match surface, hard/clay/grass level, a 270-day exponential moving average of the level (display only), closing, "
     "clutch, serve and return. The line is converted to win log-odds against a reference opponent with serve and return both at the "
     "level of the 10th/20th/50th best active player at that moment (active = 30+ rated matches and a match in the past 365 days), "
     "using the chain of 5.4–5.6 with μ<sub>Hard</sub> (or the chosen surface), plus closing and w<sub>clutch</sub>·clutch when switched on. "
     "'vs No. 5–15' averages the win probability against each of the players ranked 5–15. The Serve view holds the player's return at "
     "the reference level (and the Return view the serve). 'Raw ratings' plots the numbers themselves. Women's charts start in 1992.",
     "<b>Current ratings</b>: active players (30+ rated matches, played in the past 365 days; top 450 by level for men). "
     "<b>Strength</b> = the full model's average win probability (tour event, best of 3, standard court) against the players ranked "
     "5th–15th by level, excluding oneself; General = 0.6·hard + 0.3·clay + 0.1·grass. Serve/Return columns show s and r on the chosen surface.",
     "<b>Elo</b> for comparison: win/loss only, K = 250/(n + 5)<super>0.4</super> with n the player's prior matches in that pool, start 1500; "
     "overall and per-surface pools updated on every rated row; the surface views use 0.5·overall + 0.5·surface Elo.",
     "<b>Match odds</b>: the final prediction of section 7 for two chosen players; headline = chance to advance (retirement risk included), breakdown = chance if completed.",
     "<b>Player profiles</b>: every match (rated or not) for players with 10+ rated matches: pre-match final win probability, level "
     "before, and the change in level, serve, return and match-surface level caused by the match, plus serve and return points won. "
     "Slams and titles won against <b>predicted</b>: the sum of the player's pre-event title chances over every tour-level knockout event entered "
     "(state frozen on the eve of the event; 1,200 seeded random draws, 3,000 at Slams; seeds by official rank, byes to top seeds; match chances "
     "from 5.4–5.6 with the retirement risk of section 7). Not counted: Davis/Fed Cup, round-robin finals, men's challengers."])

# ======================================================================================= 12 reference values
H1("12. Reference values for a re-implementation")
P("With the data described in section 3, a correct re-implementation should reproduce these values (state at the last date in "
  "the data, layoff decay applied).")
TBL([["Player", "Level", "Serve S", "Return R", "Closing c", "Clutch"],
     ["Jannik Sinner", "1.1912", "1.4607", "0.9216", "0.3711", "0.0253"],
     ["Carlos Alcaraz", "1.0097", "0.8490", "1.1705", "0.3191", "0.0515"],
     ["Alexander Zverev", "0.9632", "1.2534", "0.6730", "0.3648", "0.0048"],
     ["Iga Swiatek", "0.9911", "0.9389", "1.0432", "0.2675", "−0.0275"],
     ["Coco Gauff", "0.9617", "0.9002", "1.0231", "0.2593", "0.0525"]], [44, 24, 24, 24, 26, 26])
TBL([["Surface gaps (serve / return)", "Hard", "Clay", "Grass"],
     ["Sinner", "0.0961 / 0.0756", "−0.1227 / −0.0563", "−0.2085 / −0.2847"],
     ["Swiatek", "0.0165 / −0.0018", "0.0578 / 0.0660", "−0.2725 / −0.1872"]], [50, 40, 40, 40])
TBL([["Surface averages μ", "Hard", "Clay", "Grass", "Carpet"],
     ["Men", "0.5434", "0.4282", "0.6039", "0.5683"],
     ["Women", "0.2769", "0.2785", "0.4509", "0.5391"]], [50, 30, 30, 30, 30])
TBL([["Final probability to advance (neutral context, retirement risk included)", "P(first advances)"],
     ["Sinner v Alcaraz, hard, best of 3, tour event, standard court", "0.6536"],
     ["Sinner v Alcaraz, clay, best of 5, Grand Slam, speed −0.2", "0.6044"],
     ["Sinner v Alcaraz, grass, best of 3, Masters, speed +0.15", "0.6403"],
     ["Swiatek v Gauff, hard, best of 3, tour event, standard court", "0.5402"],
     ["Swiatek v Gauff, grass, best of 3, WTA 1000, speed +0.15", "0.4766"]], [130, 40])
P("Held-out logloss after refitting the corrections (completed matches): men 0.60960 (accuracy 66.10%), women 0.59097 (68.06%).")

# ======================================================================================= 13 limitations
H1("13. Limitations and open ideas")
BUL(["The serve/return split does not improve win odds; its value is interpretability and, potentially, score-level markets "
     "(tiebreak chances, number of breaks, totals), which have not been modelled.",
     "Women's data has no ITF or qualifying matches, so newcomers and lower-ranked players are rated on fewer matches.",
     "Before 1991 (men) and before the mid-2000s (women) ratings learn from games only; ratings in those eras are noisier.",
     "In women's tennis the server's rating carries about 10% more weight than the returner's in a free regression (men: equal); "
     "part of that is measurement noise. A separate return weight in the point model is untested.",
     "Ratings react to points, not wins: a player who wins many close matches while being out-pointed rises slowly; the clutch and "
     "closing terms capture part of that.",
     "Court speed relies on ace counts and is unavailable for events without history; the feature is then zero."])

doc = Doc(PATHS.OUT + "/tennis_model_documentation.pdf")
doc.multiBuild(story)
print("ok")
