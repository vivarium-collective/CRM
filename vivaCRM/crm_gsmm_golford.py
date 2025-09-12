from process_bigraph import Process
from process_bigraph.emitter import emitter_from_wires
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from process_bigraph import Process
import numpy as np
from scipy.integrate import solve_ivp

class crm_goldford(Process):
    """
    Goldford-style Consumer-Resource Process (STRICT = standalone match)

    Config (dict):
      species_names: [string]
      resource_names: [string]  (or key 'resources_names')
      yields: {species: {resource: float}}                    # gDW / mmol
      resource_uptakes: {species: {resource: float}}          # Vmax (mmol/(gDW·h))
      maintenance: {species: float}                           # 1/h
      Km: {resource: float}                                   # mmol/L (vector ONLY)
      byproducts: {species: {consumed: {produced: float}}}    # produced_i per consumed_j (mmol/mmol)
      dilution: float                                         # 1/h (0 for batch)
      feed: {resource: float}                                 # mmol/L (used iff dilution>0)
      resource_loss: {resource: float}                        # 1/h
    """

    config_schema = {
        "species_names": "any",
        "resources_names": "any",        # optional alias
        "resource_names": "any",
        "yields": "map[map[float]]",
        "maintenance": "map[float]",
        "Km": "map[float]",              # STRICT: vector form only
        "byproducts": "map[map[float]]",
        "dilution": "float",
        "feed": "map[float]",
        "resource_loss": "map[float]",
        "resource_uptakes": "map[map[float]]"
    }

    # --- constructor MUST pass config to base class; do not call initialize() yourself
    def __init__(self, config, core=None, **kwargs):
        super().__init__(config, core=core, **kwargs)

    def initialize(self, config):
        # Names & indices
        self.species_names = list(config["species_names"])
        self.resource_names = list(config.get("resource_names", config.get("resources_names")))
        self.S, self.R = len(self.species_names), len(self.resource_names)
        self.s_idx = {s: i for i, s in enumerate(self.species_names)}
        self.r_idx = {r: j for j, r in enumerate(self.resource_names)}

        # Maps
        Y_map    = dict(config.get("yields", {}))
        Vmax_map = dict(config.get("resource_uptakes", {}))
        maint_map= dict(config.get("maintenance", {}))
        Km_map   = dict(config.get("Km", {}))      # STRICT vector
        B_map    = dict(config.get("byproducts", {}))
        self.delta = float(config.get("dilution", 0.0))
        feed_map = dict(config.get("feed", {}))
        loss_map = dict(config.get("resource_loss", {}))

        # Arrays
        self.Y = np.zeros((self.S, self.R))
        self.V = np.zeros((self.S, self.R))
        self.B = np.zeros((self.S, self.R, self.R))
        self.maint = np.zeros(self.S)
        self.Km = np.zeros(self.R)                 # STRICT vector
        self.feed = np.zeros(self.R) if self.delta > 0 else None
        self.loss = np.zeros(self.R)

        # Fill Y, Vmax, maintenance
        for s, si in self.s_idx.items():
            self.maint[si] = float(maint_map.get(s, 0.02))
            y_row = Y_map.get(s, {})
            v_row = Vmax_map.get(s, {})
            for r, rj in self.r_idx.items():
                self.Y[si, rj] = float(y_row.get(r, 0.0))
                self.V[si, rj] = float(v_row.get(r, 0.0))

        # Byproducts: produced-per-consumed
        for s, si in self.s_idx.items():
            s_block = B_map.get(s, {})
            if not isinstance(s_block, dict):
                continue
            for consumed, produceds in s_block.items():
                j = self.r_idx.get(consumed)
                if j is None or not isinstance(produceds, dict):
                    continue
                for produced, val in produceds.items():
                    i = self.r_idx.get(produced)
                    if i is not None:
                        self.B[si, i, j] = float(val)

        # Km: enforce vector form like the standalone
        # (fails fast if user accidentally passed nested per-species Km)
        for r, rj in self.r_idx.items():
            if isinstance(Km_map.get(r, 0.5), dict):
                raise ValueError("crm_goldford: Km must be a {resource: float} map to match standalone.")
            self.Km[rj] = float(Km_map.get(r, 0.5))

        # Feed & loss
        if self.delta > 0:
            for r, rj in self.r_idx.items():
                self.feed[rj] = float(feed_map.get(r, 0.0))
        for r, rj in self.r_idx.items():
            self.loss[rj] = float(loss_map.get(r, 0.0))

    # --- dynamics: EXACT standalone formulas
    def mcrm_ode(self, t, y):
        S, R = self.S, self.R
        n = y[:S]
        Rconc = y[S:S+R]

        # u = R / (Km + R)
        u = Rconc / (self.Km + Rconc + 1e-12)      # (R,)
        v = self.V * u[None, :]                    # (S,R)
        growth = (self.Y * v).sum(axis=1)          # (S,)

        dn_dt = n * (growth - self.maint - self.delta)
        cons  = (n[:, None] * v).sum(axis=0)       # (R,)
        produced = (n[:, None, None] * self.B * v[:, None, :]).sum(axis=(0, 2))  # (R,)

        feed_term = (self.delta * (self.feed - Rconc)) if (self.delta > 0) else np.zeros(R)
        loss_term = self.loss * Rconc              # EXACT: no clamp

        dR_dt = feed_term - cons + produced - loss_term
        return np.concatenate([dn_dt, dR_dt])

    def inputs(self):
        return {
            "species": "map[float]",
            "concentrations": "map[float]",
            "strategies": "map[map[float]]",
        }

    def outputs(self):
        return {
            "species_delta": "map[float]",
            "concentrations_delta": "map[float]",
            "strategies_delta": "map[map[float]]",
        }

    def update(self, state, interval):
        # Read state
        n0 = np.array([state["species"].get(s, 0.0) for s in self.species_names], dtype=float)
        R0 = np.array([state["concentrations"].get(r, 0.0) for r in self.resource_names], dtype=float)
        y0 = np.concatenate([n0, R0])

        sol = solve_ivp(self.mcrm_ode, [0.0, float(interval)], y0, t_eval=[float(interval)], method="RK45")
        if not sol.success:
            raise RuntimeError(sol.message)
        y1 = sol.y[:, -1]                           # EXACT: no clamping

        n1 = y1[:self.S]
        R1 = y1[self.S:self.S + self.R]

        # Deltas
        delta_species = {s: float(n1[i] - n0[i]) for i, s in enumerate(self.species_names)}
        delta_conc    = {r: float(R1[j] - R0[j]) for j, r in enumerate(self.resource_names)}

        # Strategies ignored (but keep schema)
        strategies_in = state.get("strategies", {}) or {}
        strategies_delta = {s: {rr: 0.0 for rr in (strategies_in.get(s, {}) or self.resource_names)}
                            for s in self.species_names}

        return {
            "species_delta": delta_species,
            "concentrations_delta": delta_conc,
            "strategies_delta": strategies_delta
        }


# ---------- convenience: emitter wiring ----------
def get_crm_goldford_emitter(state_keys):
    """
    Returns a standard emitter step spec for CRM Goldford simulations.
    Includes only the state keys present.
    """
    POSSIBLE_KEYS = {"species", "concentrations", "strategies", "global_time"}
    included = [k for k in POSSIBLE_KEYS if k in state_keys]
    return emitter_from_wires({k: [k] for k in included})


# ---------- convenience: plotting ----------
def plot_crm_goldford_simulation(results):
    """
    Plots species and resource dynamics (and strategies if present)
    from gather_emitter_results(sim)[('emitter',)].
    """
    if not results:
        raise ValueError("No results to plot")

    first = results[0]
    species_names = list(first.get("species", {}).keys())
    resource_names = list(first.get("concentrations", {}).keys())
    has_strategies = "strategies" in first
    times = np.array([r.get("global_time", i) for i, r in enumerate(results)])

    biomass = np.array([[r["species"][s] for s in species_names] for r in results])
    resources = np.array([[r["concentrations"][res] for res in resource_names] for r in results])

    n_rows = 3 if has_strategies else 2
    fig, axs = plt.subplots(n_rows, 1, figsize=(8, 7), sharex=True)

    # Biomass
    for i, s in enumerate(species_names):
        axs[0].plot(times, biomass[:, i], label=s)
    axs[0].set_ylabel("Biomass [gDW/L]")
    axs[0].set_title("Species Dynamics")
    axs[0].legend()

    # Resources
    for j, rname in enumerate(resource_names):
        axs[1].plot(times, resources[:, j], label=rname)
    axs[1].set_ylabel("Concentration [mmol/L]")
    axs[1].set_title("Resource Dynamics")
    axs[1].legend()

    # Strategies (if present)
    if has_strategies:
        # Flatten strategy traces per species-resource
        for s in species_names:
            mat = np.array([[r["strategies"][s][res] for res in resource_names] for r in results])
            for j, rname in enumerate(resource_names):
                axs[2].plot(times, mat[:, j], label=f"{s}-{rname}")
        axs[2].set_ylabel("Strategy")
        axs[2].set_xlabel("Time (h)")
        axs[2].set_title("Strategy (ignored by model; deltas=0)")
        axs[2].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


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