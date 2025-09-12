from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from process_bigraph import Process
from scipy.integrate import solve_ivp

# ---------- params ----------
@dataclass
class MCRMParams:
    yields: np.ndarray            # (m,p)  gDW/mmol
    vmax: np.ndarray              # (m,p)  mmol/(gDW·h)
    B: np.ndarray                 # (m,p,p) produced_i per consumed_j
    Km: np.ndarray                # (p,)   mmol/L
    maintenance: np.ndarray       # (m,)   1/h
    dilution: float = 0.0         # 1/h
    feed: Optional[np.ndarray] = None       # (p,)
    resource_loss: Optional[np.ndarray] = None  # (p,)

    def validate(self) -> None:
        m, p = self.yields.shape
        assert self.vmax.shape == (m, p)
        assert self.B.shape == (m, p, p)
        assert self.Km.shape == (p,)
        assert self.maintenance.shape == (m,)
        if self.dilution != 0.0:
            assert self.feed is not None and self.feed.shape == (p,)
        if self.resource_loss is not None:
            assert self.resource_loss.shape == (p,)


# ---------- CRM Ode ----------
def _monod_u(R: np.ndarray, Km: np.ndarray) -> np.ndarray:
    return R / (Km + R + 1e-12)

def mcrm_rhs(t: float, x: np.ndarray, params: MCRMParams) -> np.ndarray:
    params.validate()
    m, p = params.yields.shape
    n = x[:m]
    R = x[m:]

    u = _monod_u(R, params.Km)              # (p,)
    v = params.vmax * u[None, :]            # (m,p)
    growth = (params.yields * v).sum(axis=1)
    dn = n * (growth - params.maintenance - params.dilution)

    consumption = (n[:, None] * v).sum(axis=0)    # (p,)
    produced = np.zeros(p)
    for s in range(m):
        produced += n[s] * (params.B[s] @ v[s])

    feed_term = np.zeros(p) if params.dilution == 0.0 else params.dilution * (params.feed - R)
    loss_term = np.zeros(p) if params.resource_loss is None else params.resource_loss * R
    dR = feed_term - consumption + produced - loss_term

    return np.concatenate([dn, dR])
#%%
def build_params(
    species_names: List[str],
    resource_names: List[str],
    *,
    yields_map: Dict[str, Dict[str, float]],
    vmax_map: Dict[str, Dict[str, float]],
    maintenance_map: Dict[str, float],
    Km_map: Dict[str, float],
    byproducts_map: Dict[str, Dict[str, Dict[str, float]]],
    dilution: float = 0.0,
    feed_map: Optional[Dict[str, float]] = None,
    resource_loss_map: Optional[Dict[str, float]] = None,
    byproduct_sign: str = "as_is",  # "as_is" | "abs" | "neg_to_pos"
) -> MCRMParams:
    m, p = len(species_names), len(resource_names)
    s_idx = {s: i for i, s in enumerate(species_names)}
    r_idx = {r: j for j, r in enumerate(resource_names)}

    Y  = np.zeros((m, p))
    V  = np.zeros((m, p))
    B  = np.zeros((m, p, p))
    Km = np.zeros(p)
    M  = np.zeros(m)

    feed = None
    if dilution > 0.0:
        feed = np.zeros(p)
        if feed_map:
            for r, j in r_idx.items():
                feed[j] = float(feed_map.get(r, 0.0))

    loss = None
    if resource_loss_map is not None:
        loss = np.zeros(p)
        for r, j in r_idx.items():
            loss[j] = float(resource_loss_map.get(r, 0.0))

    for s, si in s_idx.items():
        M[si] = float(maintenance_map.get(s, 0.02))
        yrow = yields_map.get(s, {}) or {}
        vrow = vmax_map.get(s, {}) or {}
        for r, rj in r_idx.items():
            Y[si, rj] = float(yrow.get(r, 0.0))
            V[si, rj] = float(vrow.get(r, 0.0))

    for s, si in s_idx.items():
        for consumed, produceds in (byproducts_map.get(s, {}) or {}).items():
            j = r_idx.get(consumed)
            if j is None: continue
            for produced, val in (produceds or {}).items():
                i = r_idx.get(produced)
                if i is None: continue
                raw = float(val)
                if byproduct_sign == "abs": stoich = abs(raw)
                elif byproduct_sign == "neg_to_pos": stoich = -raw
                else: stoich = raw
                B[si, i, j] = stoich

    for r, rj in r_idx.items():
        Km[rj] = float(Km_map.get(r, 0.5))

    params = MCRMParams(Y, V, B, Km, M, float(dilution), feed, loss)
    params.validate()
    return params

#%%
class MCRM_Process(Process):
    """
    Minimal Goldford-MCRM process for process_bigraph.
    Ports:
      inputs : species (map[float]), concentrations (map[float])
      outputs: species_delta (map[float]), concentrations_delta (map[float])
    """

    config_schema = {
        "species_names": "list[string]",
        "resource_names": "list[string]",
        "params": "any",  # pass MCRMParams or dict-of-arrays
        # optional solver knobs:
        "method": {"_type": "string",  "_default": "RK45"},
        "rtol":   {"_type": "float",   "_default": 1e-5},
        "atol":   {"_type": "float",   "_default": 1e-7},
        "max_step": {"_type": "float", "_default": 0.05},
        "clip_nonnegative": {"_type": "boolean", "_default": True},
    }

    def _coerce_params(self, p):
        if isinstance(p, MCRMParams):
            return p
        if isinstance(p, dict):
            feed = p.get("feed", None); loss = p.get("resource_loss", None)
            return MCRMParams(
                yields=np.asarray(p["yields"], float),
                vmax=np.asarray(p["vmax"], float),
                B=np.asarray(p["B"], float),
                Km=np.asarray(p["Km"], float),
                maintenance=np.asarray(p["maintenance"], float),
                dilution=float(p.get("dilution", 0.0)),
                feed=None if feed is None else np.asarray(feed, float),
                resource_loss=None if loss is None else np.asarray(loss, float),
            )
        raise ValueError("config['params'] must be MCRMParams or dict with arrays")

    def initialize(self, config):
        self.species_names  = list(config["species_names"])
        self.resource_names = list(config["resource_names"])
        self.params = self._coerce_params(config["params"])
        self.params.validate()

        m, p = self.params.yields.shape
        if len(self.species_names) != m or len(self.resource_names) != p:
            raise ValueError("names length must match array shapes")

        allowed = {"rk23":"RK23","rk45":"RK45","dop853":"DOP853","radau":"Radau","bdf":"BDF","lsoda":"LSODA"}
        self.method = allowed.get(str(config.get("method","LSODA")).lower(), "LSODA")
        self.rtol   = float(config.get("rtol", 1e-5))
        self.atol   = float(config.get("atol", 1e-7))
        self.max_step = float(config.get("max_step", 0.05))
        self.clip   = bool(config.get("clip_nonnegative", True))

        self.m, self.p = m, p

    def inputs(self):
        return {"species": "map[float]", "concentrations": "map[float]"}

    def outputs(self):
        return {"species_delta": "map[float]", "concentrations_delta": "map[float]"}

    def _ode(self, t, y):
        # inline rhs identical to mcrm_rhs
        m, p = self.m, self.p
        n = y[:m]; R = y[m:]

        Y, Vmax, B, Km = self.params.yields, self.params.vmax, self.params.B, self.params.Km
        maint, delta, feed, mu = self.params.maintenance, self.params.dilution, self.params.feed, self.params.resource_loss

        R = np.maximum(R, 0.0)
        u = R / (Km + R + 1e-12)
        v = Vmax * u[None, :]

        growth = (Y * v).sum(axis=1)
        dn_dt  = n * (growth - maint - delta)

        cons = (n[:, None] * v).sum(axis=0)
        produced = np.zeros(p)
        for s in range(m):
            produced += n[s] * (B[s] @ v[s])

        feed_term = np.zeros(p) if delta == 0.0 else delta * (feed - R)
        loss_term = np.zeros(p) if mu is None else mu * R
        dR_dt = feed_term - cons + produced - loss_term

        return np.concatenate([dn_dt, dR_dt])

    def update(self, state, interval):
        n0 = np.array([state["species"].get(s, 0.0) for s in self.species_names], float)
        R0 = np.array([state["concentrations"].get(r, 0.0) for r in self.resource_names], float)
        y0 = np.concatenate([n0, R0])

        sol = solve_ivp(self._ode, [0.0, float(interval)], y0,
                        method=self.method, rtol=self.rtol, atol=self.atol,
                        max_step=self.max_step, t_eval=[float(interval)])
        if not sol.success:
            raise RuntimeError(sol.message)

        y1 = sol.y[:, -1]
        if self.clip: y1 = np.maximum(y1, 0.0)

        n1, R1 = y1[:self.m], y1[self.m:]
        return {
            "species_delta": {s: float(n1[i] - n0[i]) for i, s in enumerate(self.species_names)},
            "concentrations_delta": {r: float(R1[j] - R0[j]) for j, r in enumerate(self.resource_names)},
        }