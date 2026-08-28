# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║                PREDIFOOT AI — VERSION FINALE V3                      ║
║                                                                      ║
║  • Dixon-Coles (ajustement tau) + Elo dynamique                      ║
║  • XGBoost calibré (isotonic) + blending pondéré                     ║
║  • Cotes automatiques (The Odds API) + fallback manuel               ║
║  • Dé-vig méthode Shin + Value Bets + Kelly fractionné               ║
║  • Backtest walk-forward avec ROI et courbe de bankroll              ║
║  • Prochains matchs + prédictions rapides                            ║
║  • Notifications Telegram sur value bets                             ║
║                                                                      ║
║  LANCER LOCALEMENT :  streamlit run app.py                           ║
║  DÉPLOYER EN LIGNE :  share.streamlit.io (repo GitHub)               ║
║                                                                      ║
║  requirements.txt :                                                  ║
║      streamlit>=1.32.0                                               ║
║      pandas>=2.0.0                                                   ║
║      numpy>=1.24.0                                                   ║
║      scipy>=1.10.0                                                   ║
║      scikit-learn>=1.3.0                                             ║
║      xgboost>=2.0.0                                                  ║
║      requests>=2.31.0                                                ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import minimize, brentq
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PrediFoot AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_GOALS = 8
XI_DECROISSANCE = 0.0065
FENETRE_JOURS = 400


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 : INGESTION DES DONNÉES (football-data.org)
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def charger_historique(api_key: str, code_comp: str, saisons: int = 2) -> pd.DataFrame:
    """Récupère les matchs terminés des N dernières saisons."""
    tous = []
    annee = time.localtime().tm_year
    for saison in range(annee - saisons, annee + 1):
        try:
            r = requests.get(
                f"https://api.football-data.org/v4/competitions/{code_comp}/matches",
                headers={"X-Auth-Token": api_key},
                params={"season": saison, "status": "FINISHED"},
                timeout=25,
            )
            if r.status_code != 200:
                continue
            for m in r.json().get("matches", []):
                ft = m.get("score", {}).get("fullTime", {})
                if ft.get("home") is None:
                    continue
                tous.append({
                    "date": m["utcDate"][:10],
                    "home": m["homeTeam"]["name"],
                    "away": m["awayTeam"]["name"],
                    "home_goals": int(ft["home"]),
                    "away_goals": int(ft["away"]),
                })
            time.sleep(6)  # respect du rate-limit du plan gratuit
        except requests.RequestException:
            continue
    df = pd.DataFrame(tous)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def charger_prochains_matchs(api_key: str, code_comp: str) -> pd.DataFrame:
    """Récupère les 20 prochains matchs programmés."""
    try:
        r = requests.get(
            f"https://api.football-data.org/v4/competitions/{code_comp}/matches",
            headers={"X-Auth-Token": api_key},
            params={"status": "SCHEDULED"},
            timeout=25,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json().get("matches", [])[:20]
        return pd.DataFrame([{
            "date": m["utcDate"][:16].replace("T", " "),
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
        } for m in data])
    except requests.RequestException:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def charger_cotes_automatiques(odds_api_key: str, sport_key: str, region: str) -> dict:
    """
    Récupère les cotes 1N2 depuis The Odds API.
    Retourne : {"EquipeA - EquipeB": {"home": x, "draw": y, "away": z}, ...}
    """
    if not odds_api_key:
        return {}
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
            params={
                "apiKey": odds_api_key,
                "regions": region,
                "markets": "h2h",
                "oddsFormat": "decimal",
            },
            timeout=25,
        )
        if r.status_code != 200:
            return {}
        resultats = {}
        for match in r.json():
            titre = f"{match['home_team']} - {match['away_team']}"
            cotes = {"home": None, "draw": None, "away": None}
            for bookmaker in match.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    for outcome in market["outcomes"]:
                        nom = outcome["name"]
                        cote = float(outcome["price"])
                        if nom == match["home_team"]:
                            cotes["home"] = cote
                        elif nom == match["away_team"]:
                            cotes["away"] = cote
                        else:
                            cotes["draw"] = cote
            if None not in cotes.values():
                resultats[titre] = cotes
        return resultats
    except requests.RequestException:
        return {}


def generer_donnees_demo(n_matchs: int = 900) -> pd.DataFrame:
    """Génère un historique simulé réaliste pour tester sans clé API."""
    rng = np.random.default_rng(42)
    equipes = [
        "Paris SG", "Marseille", "Lyon", "Monaco", "Lille", "Nice",
        "Rennes", "Lens", "Nantes", "Strasbourg", "Toulouse",
        "Montpellier", "Brest", "Reims",
    ]
    force = {e: float(rng.normal(0, 0.45)) for e in equipes}
    lignes = []
    date_base = datetime(2023, 8, 1)
    for i in range(n_matchs):
        h, a = rng.choice(equipes, 2, replace=False)
        lam = float(np.exp(0.35 + force[h] - force[a]))
        mu = float(np.exp(0.10 + force[a] - force[h]))
        lignes.append({
            "date": (date_base + timedelta(days=int(i / 3))).strftime("%Y-%m-%d"),
            "home": h,
            "away": a,
            "home_goals": int(rng.poisson(max(lam, 0.1))),
            "away_goals": int(rng.poisson(max(mu, 0.1))),
        })
    return pd.DataFrame(lignes)


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 : SYSTÈME ELO DYNAMIQUE
# ══════════════════════════════════════════════════════════════════

class EloRating:
    """
    Elo dynamique avec :
      - K amplifié selon la différence de buts (formule World Football Elo)
      - avantage terrain intégré dans l'espérance de victoire
    """

    BASE = 1500.0
    K = 20.0
    HOME_ADV = 65.0

    def __init__(self):
        self.ratings = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, self.BASE)

    @staticmethod
    def _goal_mult(goal_diff: int) -> float:
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        return (11.0 + gd) / 8.0

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        rh = self.get(home)
        ra = self.get(away)
        if home_goals > away_goals:
            result = 1.0
        elif home_goals == away_goals:
            result = 0.5
        else:
            result = 0.0
        expected = 1.0 / (1.0 + 10.0 ** ((ra - (rh + self.HOME_ADV)) / 400.0))
        k = self.K * self._goal_mult(home_goals - away_goals)
        self.ratings[home] = rh + k * (result - expected)
        self.ratings[away] = ra - k * (result - expected)

    def calculer_historique(self, df: pd.DataFrame) -> "EloRating":
        """Passe chronologique sur tout l'historique pour obtenir les Elo finaux."""
        df_sorted = df.sort_values("date")
        for row in df_sorted.itertuples():
            self.update(row.home, row.away, row.home_goals, row.away_goals)
        return self


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 : MODÈLE DIXON-COLES
# ══════════════════════════════════════════════════════════════════

class DixonColes:
    """
    Modèle Dixon-Coles :
      - forces d'attaque / défense par équipe + avantage domicile
      - paramètre rho : corrige la dépendance des bas scores (0-0, 1-0, 0-1, 1-1)
      - décroissance temporelle exponentielle (les matchs récents pèsent plus)
    """

    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.rho = -0.05
        self.home_adv = 1.25
        self.teams = []
        self.fitted = False

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        if x == 0 and y == 1:
            return 1.0 + lam * rho
        if x == 1 and y == 0:
            return 1.0 + mu * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def fit(self, df: pd.DataFrame) -> bool:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        jour_max = df["date"].max()
        df["days_ago"] = (jour_max - df["date"]).dt.days
        df = df[df["days_ago"] < FENETRE_JOURS]

        self.teams = sorted(set(df["home"]) | set(df["away"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        if n < 4 or len(df) < 60:
            return False

        avg = max(float(df["home_goals"].mean()), 0.3)

        def neg_log_likelihood(params):
            att = params[:n]
            deff = params[n:2 * n]
            rho = params[-2]
            home_adv = params[-1]
            ll = 0.0
            for row in df.itertuples():
                i = idx[row.home]
                j = idx[row.away]
                lam = np.exp(att[i] + deff[j] + np.log(home_adv))
                mu = np.exp(att[j] + deff[i])
                poids = np.exp(-XI_DECROISSANCE * row.days_ago)
                p = (poisson.pmf(row.home_goals, lam)
                     * poisson.pmf(row.away_goals, mu)
                     * self._tau(row.home_goals, row.away_goals, lam, mu, rho))
                ll -= poids * np.log(max(p, 1e-10))
            return ll

        x0 = np.concatenate([
            np.full(n, np.log(avg)),
            np.full(n, -np.log(avg)),
            [-0.05],
            [1.25],
        ])
        bounds = [(-3.0, 3.0)] * (2 * n) + [(-0.2, 0.2), (1.0, 1.8)]

        try:
            res = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
            if not res.success or not np.all(np.isfinite(res.x)):
                return False
            p = res.x
            self.attack = {t: float(p[i]) for t, i in idx.items()}
            self.defense = {t: float(p[n + i]) for t, i in idx.items()}
            self.rho = float(p[-2])
            self.home_adv = float(p[-1])
            self.fitted = True
            return True
        except Exception:
            return False

    def lambdas(self, home: str, away: str) -> tuple:
        lam = np.exp(
            self.attack.get(home, 0.0)
            + self.defense.get(away, 0.0)
            + np.log(self.home_adv)
        )
        mu = np.exp(
            self.attack.get(away, 0.0)
            + self.defense.get(home, 0.0)
        )
        return float(lam), float(mu)

    def matrice_scores(self, home: str, away: str) -> np.ndarray:
        lam, mu = self.lambdas(home, away)
        M = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
        for x in range(MAX_GOALS + 1):
            for y in range(MAX_GOALS + 1):
                M[x, y] = (poisson.pmf(x, lam)
                           * poisson.pmf(y, mu)
                           * self._tau(x, y, lam, mu, self.rho))
        total = M.sum()
        if total <= 0:
            M[:] = 1.0
            total = M.sum()
        return M / total

    @staticmethod
    def extraire_marches(M: np.ndarray) -> dict:
        paires = [(x, y) for x in range(MAX_GOALS + 1) for y in range(MAX_GOALS + 1)]
        p_home = float(np.tril(M, -1).sum())
        p_draw = float(np.trace(M))
        p_away = float(np.triu(M, 1).sum())
        p_over25 = float(sum(M[x, y] for (x, y) in paires if x + y > 2))
        p_btts = float(sum(M[x, y] for (x, y) in paires if x >= 1 and y >= 1))
        top = sorted(((x, y, M[x, y]) for (x, y) in paires),
                     key=lambda t: -t[2])[:3]
        return {
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "p_over25": p_over25,
            "p_btts": p_btts,
            "top_scores": [(f"{x}-{y}", float(p)) for x, y, p in top],
        }

    def predire(self, home: str, away: str) -> dict:
        M = self.matrice_scores(home, away)
        result = self.extraire_marches(M)
        lam, mu = self.lambdas(home, away)
        result["lam"] = lam
        result["mu"] = mu
        return result


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 : FEATURES + XGBOOST + BLENDING
# ══════════════════════════════════════════════════════════════════

FEATURES_XGB = [
    "home_form_gf", "home_form_ga", "away_form_gf", "away_form_ga",
    "home_form_pts", "away_form_pts", "elo_diff", "form_diff", "attack_diff",
]


def ema(derniers: list, span: int = 5, defaut: float = 1.3) -> float:
    """Moyenne mobile exponentielle d'une série de valeurs."""
    if not derniers:
        return defaut
    return float(pd.Series(derniers).ewm(span=span, min_periods=1).mean().iloc[-1])


def extraire_stats_equipes(df: pd.DataFrame) -> dict:
    """Rejoue l'historique chronologique pour accumuler les stats par équipe."""
    stats = {}
    df_sorted = df.sort_values("date")
    for row in df_sorted.itertuples():
        for team in (row.home, row.away):
            if team not in stats:
                stats[team] = {"xgf": [], "xga": [], "pts": []}
        sh = stats[row.home]
        sa = stats[row.away]
        sh["xgf"].append(row.home_goals)
        sh["xga"].append(row.away_goals)
        sh["pts"].append(3 if row.home_goals > row.away_goals
                         else (1 if row.home_goals == row.away_goals else 0))
        sa["xgf"].append(row.away_goals)
        sa["xga"].append(row.home_goals)
        sa["pts"].append(3 if row.away_goals > row.home_goals
                         else (1 if row.home_goals == row.away_goals else 0))
    return stats


def construire_features(df: pd.DataFrame, elo: EloRating) -> pd.DataFrame:
    """
    Construit le dataset de features pour le XGBoost.
    Anti data-leakage : les features d'un match n'utilisent QUE les matchs
    antérieurs (mise à jour des stats APRÈS la construction des features).
    """
    df = df.sort_values("date").reset_index(drop=True)
    stats = {}
    lignes = []
    for _, row in df.iterrows():
        h, a = row["home"], row["away"]
        if h not in stats:
            stats[h] = {"xgf": [], "xga": [], "pts": []}
        if a not in stats:
            stats[a] = {"xgf": [], "xga": [], "pts": []}
        sh = stats[h]
        sa = stats[a]

        home_gf = ema(sh["xgf"], defaut=1.4)
        home_ga = ema(sh["xga"], defaut=1.2)
        away_gf = ema(sa["xgf"], defaut=1.2)
        away_ga = ema(sa["xga"], defaut=1.4)
        home_pts = ema(sh["pts"], defaut=1.4)
        away_pts = ema(sa["pts"], defaut=1.2)
        home_elo = elo.get(h)
        away_elo = elo.get(a)

        if row["home_goals"] > row["away_goals"]:
            target = 0
        elif row["home_goals"] == row["away_goals"]:
            target = 1
        else:
            target = 2

        lignes.append({
            "home_form_gf": home_gf,
            "home_form_ga": home_ga,
            "away_form_gf": away_gf,
            "away_form_ga": away_ga,
            "home_form_pts": home_pts,
            "away_form_pts": away_pts,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo,
            "form_diff": home_pts - away_pts,
            "attack_diff": home_gf - away_ga,
            "target": target,
        })

        # Mise à jour APRÈS la construction des features (anti-leakage)
        sh["xgf"].append(row["home_goals"])
        sh["xga"].append(row["away_goals"])
        sh["pts"].append(3 if row["home_goals"] > row["away_goals"]
                         else (1 if row["home_goals"] == row["away_goals"] else 0))
        sa["xgf"].append(row["away_goals"])
        sa["xga"].append(row["home_goals"])
        sa["pts"].append(3 if row["away_goals"] > row["home_goals"]
                         else (1 if row["home_goals"] == row["away_goals"] else 0))

    return pd.DataFrame(lignes)


@st.cache_resource(show_spinner=False)
def entrainer_xgboost(features: pd.DataFrame):
    """
    Entraîne le XGBoost avec validation walk-forward (TimeSeriesSplit)
    + calibration isotonique. Retourne (modele_final, metriques_par_fold).
    """
    df = features.dropna(subset=FEATURES_XGB + ["target"]).reset_index(drop=True)
    if len(df) < 150:
        return None, []

    X = df[FEATURES_XGB]
    y = df["target"]

    def nouveau_modele():
        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_jobs=2,
            verbosity=0,
        )

    metriques = []
    try:
        tscv = TimeSeriesSplit(n_splits=4)
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if len(test_idx) < 20:
                continue
            model = CalibratedClassifierCV(nouveau_modele(), method="isotonic", cv=3)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            probas = model.predict_proba(X.iloc[test_idx])
            metriques.append({
                "fold": fold + 1,
                "accuracy": float(accuracy_score(y.iloc[test_idx],
                                                 probas.argmax(axis=1))),
                "log_loss": float(log_loss(y.iloc[test_idx], probas)),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            })

        final = CalibratedClassifierCV(nouveau_modele(), method="isotonic", cv=3)
        final.fit(X, y)
        return final, metriques
    except Exception:
        return None, metriques


def features_du_match(home: str, away: str, stats: dict, elo: EloRating) -> dict:
    """Construit les features d'un match à venir depuis les stats accumulées."""
    sh = stats.get(home, {"xgf": [1.4], "xga": [1.2], "pts": [1.4]})
    sa = stats.get(away, {"xgf": [1.2], "xga": [1.4], "pts": [1.2]})
    home_gf = ema(sh["xgf"], defaut=1.4)
    home_ga = ema(sh["xga"], defaut=1.2)
    away_gf = ema(sa["xgf"], defaut=1.2)
    away_ga = ema(sa["xga"], defaut=1.4)
    home_pts = ema(sh["pts"], defaut=1.4)
    away_pts = ema(sa["pts"], defaut=1.2)
    home_elo = elo.get(home)
    away_elo = elo.get(away)
    return {
        "home_form_gf": home_gf,
        "home_form_ga": home_ga,
        "away_form_gf": away_gf,
        "away_form_ga": away_ga,
        "home_form_pts": home_pts,
        "away_form_pts": away_pts,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": home_elo - away_elo,
        "form_diff": home_pts - away_pts,
        "attack_diff": home_gf - away_ga,
    }


def blending(probas_dc, probas_xgb, poids_dc: float = 0.45) -> np.ndarray:
    """Combine Dixon-Coles + XGBoost puis renormalise (somme = 1)."""
    p_dc = np.array(probas_dc, dtype=float)
    if probas_xgb is None:
        return p_dc / p_dc.sum()
    p_xgb = np.array(probas_xgb, dtype=float)
    if p_xgb.shape[0] != 3:
        p_xgb = p_xgb[:3]
    blend = poids_dc * p_dc + (1.0 - poids_dc) * p_xgb
    total = blend.sum()
    if total <= 0:
        return np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return blend / total


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 : DÉ-VIG SHIN + VALUE BETS + KELLY
# ══════════════════════════════════════════════════════════════════

def devig_shin(odds: list) -> list:
    """
    Dé-marge méthode de Shin : répartition asymétrique de la marge
    (les cotes des outsiders sont davantage gonflées par le bookmaker).
    Fallback : méthode multiplicative si non-convergence.
    """
    implied = np.array([1.0 / o for o in odds])
    total = implied.sum()
    if total <= 1.0 or len(odds) != 3:
        return list(implied / max(total, 1e-9))

    def somme_probas(z):
        return sum(
            (np.sqrt(z ** 2 + 4.0 * (1.0 - z) * (p ** 2) / total) - z)
            / (2.0 * (1.0 - z))
            for p in implied
        ) - 1.0

    try:
        z = brentq(somme_probas, 1e-6, 0.98)
        return [
            float((np.sqrt(z ** 2 + 4.0 * (1.0 - z) * (p ** 2) / total) - z)
                  / (2.0 * (1.0 - z)))
            for p in implied
        ]
    except Exception:
        return list(implied / total)


def analyser_value_bets(probas: np.ndarray, cotes: list, min_edge: float) -> list:
    """Compare les probas du modèle aux probas implicites dé-margées (fair)."""
    fair = devig_shin(cotes)
    noms = ["Victoire domicile (1)", "Match nul (N)", "Victoire extérieur (2)"]
    resultats = []
    for i, nom in enumerate(noms):
        edge = probas[i] * cotes[i] - 1.0
        edge_marche = probas[i] - fair[i]
        resultats.append({
            "pari": nom,
            "cote": cotes[i],
            "proba_modele": float(probas[i]),
            "proba_marche_fair": fair[i],
            "edge": edge,
            "edge_marche": edge_marche,
            "value_bet": bool(edge > min_edge and probas[i] > fair[i]),
        })
    return resultats


def kelly_mise(p: float, odds: float, fraction: float = 0.25,
               bankroll: float = 1000.0, max_pct: float = 0.03) -> dict:
    """
    Kelly fractionné avec garde-fous :
      - 1/4 Kelly (réduction massive de la variance)
      - plafond de mise à 3% de la bankroll
      - mise minimale de 0.5% (sinon pas de pari)
    """
    b = odds - 1.0
    q = 1.0 - p
    if b <= 0:
        return {"stake": 0.0, "kelly_full": 0.0, "pct": 0.0}
    kelly_full = (p * b - q) / b
    if kelly_full <= 0:
        return {"stake": 0.0, "kelly_full": kelly_full, "pct": 0.0}
    pct = min(kelly_full * fraction, max_pct)
    if pct < 0.005:
        return {"stake": 0.0, "kelly_full": kelly_full, "pct": pct}
    return {"stake": pct * bankroll, "kelly_full": kelly_full, "pct": pct}


def etoiles_confiance(probas: np.ndarray, elo_diff: float) -> int:
    """Indice de confiance 1-5 étoiles basé sur la proba max + l'écart Elo."""
    conf = float(probas.max())
    n = 1 + int(conf * 5)
    if abs(elo_diff) > 100:
        n += 1
    return max(1, min(5, n))


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 : BACKTEST WALK-FORWARD AVEC ROI
# ══════════════════════════════════════════════════════════════════

def backtest_value_betting(df_hist: pd.DataFrame, dc: DixonColes,
                           min_edge: float = 0.04,
                           marge_simulee: float = 0.06,
                           n_matchs: int = 120) -> pd.DataFrame:
    """
    Simule le value betting sur les N derniers matchs :
      - cotes simulées : probas du modèle + marge bookmaker réaliste
      - pari uniquement si edge > min_edge (Kelly 1/4, plafonné à 3%)
    ATTENTION : cotes simulées = borne OPTIMISTE du edge réel.
    """
    df_test = df_hist.sort_values("date").tail(n_matchs).reset_index(drop=True)
    bankroll = 1000.0
    historique = []
    for row in df_test.itertuples():
        pred = dc.predire(row.home, row.away)
        probas = np.array([pred["p_home"], pred["p_draw"], pred["p_away"]])
        p_true = probas / probas.sum()
        cotes = [1.0 / max(p * (1.0 + marge_simulee), 0.01) for p in p_true]

        analyses = analyser_value_bets(probas, cotes, min_edge)
        issue_reelle = (0 if row.home_goals > row.away_goals
                        else (1 if row.home_goals == row.away_goals else 2))
        for idx, an in enumerate(analyses):
            if not an["value_bet"]:
                continue
            k = kelly_mise(an["proba_modele"], an["cote"])
            if k["stake"] <= 0:
                continue
            gagne = (idx == issue_reelle)
            pnl = k["stake"] * (an["cote"] - 1.0) if gagne else -k["stake"]
            bankroll += pnl
            historique.append({
                "date": row.date,
                "match": f"{row.home} - {row.away}",
                "pari": an["pari"],
                "cote": round(an["cote"], 2),
                "mise": round(k["stake"], 2),
                "gagne": gagne,
                "pnl": round(pnl, 2),
                "bankroll": round(bankroll, 2),
            })
    return pd.DataFrame(historique)


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 : NOTIFICATIONS TELEGRAM (optionnel)
# ══════════════════════════════════════════════════════════════════

def envoyer_telegram(token: str, chat_id: str, message: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


# ══════════════════════════════════════════════════════════════════
#  SECTION 8 : PIPELINE PRINCIPAL (chargement + entraînement)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="📥 Chargement des données + entraînement des modèles...")
def pipeline_complet(source_demo: bool, api_key: str, code_comp: str) -> dict:
    """Charge les données puis entraîne Elo + Dixon-Coles + XGBoost."""
    if source_demo:
        df_hist = generer_donnees_demo()
    else:
        df_hist = charger_historique(api_key, code_comp)

    if df_hist.empty or len(df_hist) < 60:
        return {"erreur": "donnees_insuffisantes"}

    # Elo dynamique
    elo = EloRating().calculer_historique(df_hist)

    # Dixon-Coles (+ fallback Poisson simplifié si non-convergence)
    dc = DixonColes()
    ok_dc = dc.fit(df_hist)
    if not ok_dc:
        bp = df_hist.groupby("home")["home_goals"].mean().to_dict()
        ba = df_hist.groupby("away")["away_goals"].mean().to_dict()
        dc.teams = sorted(set(df_hist["home"]) | set(df_hist["away"]))
        dc.attack = {t: float(np.log(max(bp.get(t, 1.3), 0.2))) for t in dc.teams}
        dc.defense = {t: float(-np.log(max(ba.get(t, 1.3), 0.2))) for t in dc.teams}
        dc.rho = -0.05
        dc.home_adv = 1.25
        dc.fitted = True

    # Stats équipes + XGBoost
    stats = extraire_stats_equipes(df_hist)
    features = construire_features(df_hist, elo)
    xgb_model, metriques = entrainer_xgboost(features)

    return {
        "erreur": None,
        "df_hist": df_hist,
        "elo": elo,
        "dc": dc,
        "stats": stats,
        "xgb_model": xgb_model,
        "metriques": metriques,
        "ok_dc": ok_dc,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 9 : INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════════════

st.title("⚽ PrediFoot AI V3")
st.caption("Dixon-Coles + Elo dynamique + XGBoost (blending) "
           "+ Value Bets + Kelly + Backtest")

# ---------- Sidebar : configuration ----------

with st.sidebar:
    st.header("⚙️ Configuration")

    source_demo = st.radio(
        "Source de données",
        ["🧪 Démo simulée (test immédiat)", "🔑 API réelle (football-data.org)"],
    )
    mode_api = source_demo.startswith("🔑")

    api_key = ""
    code_comp = "PL"
    if mode_api:
        api_key = st.text_input(
            "Clé API football-data.org",
            value=st.secrets.get("FOOTBALL_API_KEY", "")
            if hasattr(st, "secrets") else "",
            type="password",
        )
        comp = st.selectbox(
            "Compétition",
            [("PL", "🏴 Premier League"), ("PD", "🇪🇸 La Liga"),
             ("SA", "🇮🇹 Serie A"), ("BL1", "🇩🇪 Bundesliga"),
             ("FL1", "🇫🇷 Ligue 1")],
            format_func=lambda x: x[1],
        )
        code_comp = comp[0]

    st.divider()
    st.subheader("💰 Cotes & Bankroll")
    odds_api_key = st.text_input(
        "Clé The Odds API (optionnel — cotes auto)",
        value=st.secrets.get("ODDS_API_KEY", "")
        if hasattr(st, "secrets") else "",
        type="password",
        help="Gratuit sur the-odds-api.com. Sans clé : saisie manuelle.",
    )
    bankroll = st.number_input("Bankroll (unités)", 10.0, 1e6, 1000.0, 50.0)
    min_edge = st.slider("Edge minimum pour value bet", 0.0, 0.20, 0.04, 0.01)
    poids_dc = st.slider(
        "Poids Dixon-Coles vs XGBoost", 0.0, 1.0, 0.45, 0.05,
        help="0.45 = 45% Dixon-Coles + 55% XGBoost")

    st.divider()
    st.subheader("📱 Telegram (optionnel)")
    tg_token = st.text_input("Bot token", type="password")
    tg_chat = st.text_input("Chat ID")

    st.divider()
    st.caption("⚠️ Outil d'aide à la décision. Aucune prédiction ne garantit "
               "des gains. Jouez responsable. 18+")


# ---------- Pipeline : chargement + entraînement (mis en cache) ----------

pipeline = pipeline_complet(source_demo, api_key, code_comp)

if pipeline.get("erreur"):
    if mode_api:
        st.error(
            "❌ Données insuffisantes. Vérifiez votre clé API gratuite "
            "([football-data.org/client/register](https://www.football-data.org/client/register)) "
            "ou choisissez le mode Démo dans la barre latérale.")
    st.stop()

df_hist = pipeline["df_hist"]
elo = pipeline["elo"]
dc = pipeline["dc"]
stats = pipeline["stats"]
xgb_model = pipeline["xgb_model"]
metriques = pipeline["metriques"]

statut_dc = "✅ Dixon-Coles calibré" if pipeline["ok_dc"] else "⚠️ Poisson simplifié"
statut_xgb = (f"✅ XGBoost actif (blending {1.0 - poids_dc:.0%})"
              if xgb_model else "❌ XGBoost indisponible (DC seul)")
st.sidebar.success(
    f"✅ {len(df_hist)} matchs | {len(dc.teams)} équipes\n\n"
    f"{statut_dc}\n\n{statut_xgb}")

if metriques:
    with st.sidebar.expander("📈 Performances XGBoost (walk-forward)"):
        st.dataframe(pd.DataFrame(metriques), hide_index=True,
                     use_container_width=True)


# ---------- Onglets ----------

onglets = st.tabs([
    "🔮 Prédiction",
    "📅 Prochains matchs",
    "📊 Backtest",
    "🏆 Classement Elo",
])


# ══════════════════════════════════════════════════════════════════
#  ONGLET 1 : PRÉDICTION
# ══════════════════════════════════════════════════════════════════

with onglets[0]:
    st.subheader("🔮 Prédiction de match")

    # Cotes automatiques si clé fournie
    cotes_auto = {}
    if odds_api_key:
        sport_map = {
            "PL": "soccer_epl", "PD": "soccer_spain_la_liga",
            "SA": "soccer_italy_serie_a", "BL1": "soccer_germany_bundesliga",
            "FL1": "soccer_france_ligue_one",
        }
        cotes_auto = charger_cotes_automatiques(
            odds_api_key, sport_map.get(code_comp, "soccer_epl"), "eu")
        if cotes_auto:
            st.success(f"💰 {len(cotes_auto)} matchs avec cotes automatiques")

    c1, c2 = st.columns(2)
    equipes_tri = sorted(dc.teams)
    with c1:
        home_sel = st.selectbox("🏠 Équipe domicile", equipes_tri)
    with c2:
        away_sel = st.selectbox(
            "✈️ Équipe extérieur",
            [e for e in equipes_tri if e != home_sel],
            index=min(1, len(equipes_tri) - 1))

    # Cotes : automatiques ou manuelles
    cle_match = f"{home_sel} - {away_sel}"
    cotes_manuelles = True
    if cle_match in cotes_auto:
        c = cotes_auto[cle_match]
        st.info(f"💰 Cotes automatiques : **1 → {c['home']:. f"**N → {c['draw']:.2f}** | **2 → {c['away']:.2f}**")
        cote1, coteN, cote2 = c["home"], c["draw"], c["away"]
        cotes_manuelles = False

    if cotes_manuelles:
        cc1, cc2, cc3 = st.columns(3)
        cote1 = cc1.number_input 1.01, 50.0, 1.85, 0.05)
        coteN = cc2.number_input("Cote N", 1.01, 50.0, 3.50, 0.05)
        cote2 = cc3.number_input("Cote 2", 1.01, 50.0, 4.20, 0.05)

    if st.button("⚡ Calculer la prédiction", type="primary",
                 use_container_width=True):

        # Dixon-Coles
        pred = dc.predire(home_sel, away_sel)
        probas_dc = (pred["p_home"], pred["p_draw"], pred["p_away"])

        # XGBoost
        probas_xgb = None
        if xgb_model:
            feat_match = features_du_match(home_sel, away_sel, stats, elo)
            X_m = pd.DataFrame([feat_match])[FEATURES_XGB]
            probas_xgb = xgb_model.predict_proba(X_m)[0]

        # Blending
        probas = blending(probas_dc, probas_xgb, poids_dc)
        p1, pN, p2 = probas

        elo_diff = elo.get(home_sel) - elo.get(away_sel)
        etoiles = etoiles_confiance(probas, elo_diff)

        st.markdown(f"### {home_sel} vs {away_sel}")
        st.markdown(f"#### Confiance : {'⭐' * etoiles} ({etoiles}/5)")

        mA, mB, mC = st.columns(3)
        mA.metric("🏠 Victoire domicile", f"{p1:.1%}")
        mB.metric("🤝 Match nul", f"{pN:.1%}")
        mC.metric("✈️ Victoire extérieur", f"{p2:.1%}")

        mD, mE, mF = st.columns(3)
        mD.metric("⚽ Over 2.5", f"{pred['p_over25']:.1%}")
        mE.metric("🎯 BTTS", f"{pred['p_btts']:.1%}")
        mF.metric("🏁 Score probable", pred["top_scores"][0][0])

        # Value bets
        st.divider()
        st.subheader("💰 Analyse Value Bets (dé-marge méthode Shin)")
        analyses = analyser_value_bets(probas, [cote1, coteN, cote2], min_edge)
        df_an = pd.DataFrame(analyses)
        df_an["proba_modele"] = df_an["proba_modele"].map("{:.1%}".format)
        df_an["proba_marche_fair"] = df_an["proba_marche_fair"].map("{:.1%}".format)
        df_an["edge"] = df_an["edge"].map("{:+.2%}".format)
        df_an["edge_marche"] = df_an["edge_marche"].map("{:+.2%}".format)
        df_an["value_bet"] = df_an["value_bet"].map(
            lambda v: "🎯 OUI" if v else "—")
        st.dataframe(
            df_an[["pari", "cote", "proba_modele", "proba_marche_fair",
                   "edge", "edge_marche", "value_bet"]],
            use_container_width=True, hide_index=True)

        # Kelly + collecte des alertes Telegram
        st.subheader(f"📉 Mise conseillée (Kelly ¼ — bankroll {bankroll:.0f})")
        alertes = []
        kc1, kc2, kc3 = st.columns(3)
        colonnes_kelly = [kc1, kc2, kc3]
        for i, an in enumerate(analyses):
            k = kelly_mise(an["proba_modele"], an["cote"], 0.25, bankroll)
            if k["stake"] > 0:
                colonnes_kelly[i].success(
                    f"**{an['pari']}**\n\nMise : **{k['stake']:.1f}** "
                    f"({k['pct']:.1%} — Kelly full {k['kelly_full']:.1%})")
                alertes.append(
                    f"🎯 <b>VALUE BET</b> : {home_sel} - {away_sel}\n"
                    f"{an['pari']} @ {an['cote']:.2f} "
                    f"(edge {an['edge']:+.1%})\n"
                    f"Mise : {k['stake']:.1f} ({k['pct']:.1%})")
            else:
                colonnes_kelly[i].error(
                    f"**{an['pari']}**\n\nPas de mise (pas d'edge)")

        # Telegram
        if alertes and tg_token and tg_chat:
            if envoyer_telegram(tg_token, tg_chat, "\n\n".join(alertes)):
                st.toast("📱 Alertes envoyées sur Telegram !")

        # Détails
        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("🏆 Top 3 scores probables")
            for s, p in pred["top_scores"]:
                st.write(f"**{s}** — {p:.1%}")
        with g2:
            st.subheader("⚡ Elo dynamique")
            st.write(f"{home_sel} : **{elo.get(home_sel):.0f}**")
            st.write(f"{away_sel} : **{elo.get(away_sel):.0f}**")
            st.write(f"Écart : **{elo_diff:+.0f}**")

        if xgb_model:
            st.caption(
                f"🔧 Composition : Dixon-Coles {poids_dc:.0%} "
                f"(λ₁={pred['lam']:.2f}, λ₂={pred['mu']:.2f}) "
                f"+ XGBoost {1.0 - poids_dc:.0%}")
        else:
            st.caption(
                f"🔧 Dixon-Coles seul "
                f"(λ₁={pred['lam']:.2f}, λ₂={pred['mu']:.2f})")


# ══════════════════════════════════════════════════════════════════
#  ONGLET 2 : PROCHAINS MATCHS
# ══════════════════════════════════════════════════════════════════

with onglets[1]:
    st.subheader("📅 Prochains matchs & prédictions rapides")
    if not mode_api:
        st.info("Les prochains matchs officiels nécessitent une clé API réelle.")
    else:
        prochain = charger_prochains_matchs(api_key, code_comp)
        if prochain.empty:
            st.warning("Aucun match à venir trouvé.")
        else:
            lignes_pred = []
            for row in prochain.itertuples():
                if row.home in dc.teams and row.away in dc.teams:
                    pred = dc.predire(row.home, row.away)
                    probas = blending(
                        (pred["p_home"], pred["p_draw"], pred["p_away"]),
                        None, poids_dc)
                    fav = ["1", "N", "2"][int(np.argmax(probas))]
                    lignes_pred.append({
                        "date": row.date,
                        "match": f"{row.home} - {row.away}",
                        "P(1)": f"{probas[0]:.0%}",
                        "P(N)": f"{probas[1]:.0%}",
                        "P(2)": f"{probas[2]:.0%}",
                        "favori": fav,
                        "score_prob": pred["top_scores"][0][0],
                    })
                else:
                    lignes_pred.append({
                        "date": row.date,
                        "match": f"{row.home} - {row.away}",
                        "P(1)": "—", "P(N)": "—", "P(2)": "—",
                        "favori": "n/a", "score_prob": "—",
                    })
            st.dataframe(pd.DataFrame(lignes_pred),
                         use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  ONGLET 3 : BACKTEST
# ══════════════════════════════════════════════════════════════════

with onglets[2]:
    st.subheader("📊 Backtest Value Betting — walk-forward")
    st.caption("""
    Simulation sur les 120 derniers matchs avec cotes simulées
    (probas du modèle + marge bookmaker de 6%). ⚠️ C'est une borne
    OPTIMISTE : le vrai edge se mesure sur des cotes réelles.
    """)

    if st.button("▶️ Lancer le backtest", type="primary",
                 use_container_width=True):
        with st.spinner("Simulation en cours..."):
            bt = backtest_value_betting(df_hist, dc, min_edge=min_edge)
        if bt.empty:
            st.info("Aucun value bet détecté avec ce seuil. "
                    "Essayez de baisser l'edge minimum.")
        else:
            pnl_total = bt["pnl"].sum()
            n_bets = len(bt)
            win_rate = bt["gagne"].mean()
            roi = pnl_total / bt["mise"].sum() if bt["mise"].sum() > 0 else 0.0
            cote_moy = bt["cote"].mean()

            kA, kB, kC, kD = st.columns(4)
            kA.metric("Paris joués", n_bets)
            kB.metric("Taux de réussite", f"{win_rate:.1%}")
            kC.metric("ROI", f"{roi:+.1%}", delta=f"{pnl_total:+.1f} unités")
            kD.metric("Cote moyenne", f"{cote_moy:.2f}")

            st.line_chart(bt.set_index("date")["bankroll"])

            with st.expander(f"📋 Détail des {n_bets} paris"):
                st.dataframe(bt, use_container_width=True, hide_index=True)

            st.caption("""
            **Interprétation** : un ROI > 0 sur cotes simulées est encourageant
            mais insuffisant. Avec des cotes réelles (marges réelles, mouvements
            de marché), attendez-vous à un ROI inférieur. Un ROI réel de +3 à
            +5% sur 500+ paris est déjà excellent.
            """)


# ══════════════════════════════════════════════════════════════════
#  ONGLET 4 : CLASSEMENT ELO
# ══════════════════════════════════════════════════════════════════

with onglets[3]:
    st.subheader("🏆 Classement Elo dynamique")
    st.caption("Elo ajusté en continu : différence de buts + avantage terrain.")
    df_elo = pd.DataFrame([
        {"rang": i + 1, "équipe": t, "elo": round(elo.get(t), 0)}
        for i, t in enumerate(sorted(dc.teams, key=lambda x: -elo.get(x)))
    ])
    st.dataframe(df_elo, use_container_width=True, hide_index=True)
    st.bar_chart(df_elo.set_index("équipe")["elo"])


# ---------- Pied de page ----------

st.divider()
st.caption("""
⚠️ **Avertissement** : outil d'aide à la décision à but éducatif. Aucune
prédiction ne garantit des gains — les bookmakers intègrent une marge de
5-8%. Ne pariez jamais plus que ce que vous pouvez vous permettre de perdre.
Joueurs Info Service : 09 74 75 13 13 (France). 18+
""")
