from process_bigraph import Process, Step
from process_bigraph import Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires
from cdFBA.utils import *
import numpy as np
from scipy.integrate import solve_ivp

def _monod(R: np.ndarray, Km: np.ndarray) -> np.ndarray:
    return R / (Km + R + 1e-12)

class GoldfordCRM(Process):
    """
    A single-step Goldford-style MCRM integrator that reads/writes the shared environment.

    Config
    ------
    species_name: str
        Key for the focal species in shared_environment["counts"].
    resource_order: list[str]
        Resource IDs (must match keys in shared_environment["counts"]).
    Km: list[float] of length p
    yields: list[float] of length p         # gDW/mmol per resource (single species)
    vmax: list[float] of length p           # mmol/(gDW·h) per resource
    B: list[list[float]] shape (p,p)
        Byproduct stoichiometry: produced_i per consumed_j.
    maintenance: float                      # 1/h
    dilution: float = 0.0                   # 1/h (chemostat); uses feed if >0
    feed: optional list[float] length p     # mmol/L feed concentrations
    resource_loss: optional list[float] p   # 1/h extra loss

    Notes
    -----
    - Uses shared_environment["volume"] to convert between counts and concentrations.
    - Produces a `crm_update` map (same shape as dFBA’s `dfba_update`), so you can
      merge with dFBA via a collector and feed into UpdateEnvironment.
    """

    config_schema = {
        "species_name": "string",
        "resource_order": "list[string]",
        "Km": "list[float]",
        "yields": "list[float]",
        "vmax": "list[float]",
        "B": "list[list[float]]",
        "maintenance": "float",
        "dilution": {"_type": "float", "_default": 0.0},
        "feed": {"_type": "maybe[list[float]]"},
        "resource_loss": {"_type": "maybe[list[float]]"},
    }

    def initialize(self, config):
        # shapes (single species)
        self.sp = config["species_name"]
        self.res_ids = list(config["resource_order"])
        p = len(self.res_ids)

        self.Km = np.array(config["Km"], float)
        self.Y  = np.array(config["yields"], float).reshape(1, p)
        self.V  = np.array(config["vmax"],   float).reshape(1, p)
        self.B  = np.array(config["B"],      float).reshape(1, p, p)
        self.m  = np.array([config["maintenance"]], float)
        self.delta = float(config.get("dilution", 0.0))

        self.feed = None
        if config.get("feed") is not None:
            self.feed = np.array(config["feed"], float)

        self.mu_loss = None
        if config.get("resource_loss") is not None:
            self.mu_loss = np.array(config["resource_loss"], float)

    def inputs(self):
        # We only need shared_environment to step the ODE
        return {
            "shared_environment": "volumetric",  # expects 'counts' + 'volume'
        }

    def outputs(self):
        # Output the same kind of update map as dFBA -> can be merged and fed to UpdateEnvironment
        return {
            "crm_update": "map[set_float]"
        }

    # ODE RHS (single-species)
    def _rhs(self, t, x, p):
        n, R = x[0], x[1:]    # n scalar, R (p,)
        u   = _monod(R, p["Km"])
        v   = p["V"][0] * u
        growth = (p["Y"][0] * v).sum()

        dn = n * (growth - p["m"][0] - p["delta"])

        consumption = n * v
        produced = (p["B"][0] @ v) * n

        if p["delta"] == 0.0:
            feed_term = 0.0 * R
        else:
            feed_term = p["delta"] * (p["feed"] - R)
        loss_term = (0.0 * R) if (p["mu_loss"] is None) else (p["mu_loss"] * R)

        dR = feed_term - consumption + produced - loss_term
        return np.concatenate([[dn], dR])

    def update(self, inputs, interval):
        env = inputs["shared_environment"]
        counts = env["counts"]
        Vsys = env["volume"]

        # Read state from counts
        n0 = float(counts[self.sp])
        R0_counts = np.array([counts[r] for r in self.res_ids], float)
        R0 = R0_counts / max(Vsys, 1e-12)  # mmol/L (or whatever your unit is)

        # Pack params for the integrator
        P = {
            "Km": self.Km, "Y": self.Y, "V": self.V, "B": self.B,
            "m": self.m, "delta": self.delta, "feed": self.feed, "mu_loss": self.mu_loss
        }

        x0 = np.concatenate([[n0], R0])
        sol = solve_ivp(lambda t, x: self._rhs(t, x, P),
                        (0.0, float(interval)), x0, t_eval=[float(interval)],
                        method="RK45", rtol=1e-7, atol=1e-9)
        xF = np.maximum(sol.y[:, -1], 0.0)
        nF, RF = xF[0], xF[1:]

        # Build the dfba-like update map (deltas)
        state_update = {}
        state_update[self.sp] = nF - n0
        # Resources are stored as counts, so convert back from concentration
        RF_counts = RF * Vsys
        dR_counts = RF_counts - R0_counts
        for rid, dval in zip(self.res_ids, dR_counts):
            state_update[rid] = float(dval)

        return {"crm_update": state_update}


class CollectSpeciesUpdates(Step):
    """
    Merge multiple per-species update maps into the 'species_updates' port expected by UpdateEnvironment.

    Config:
      sources: list[str]   # names of input ports to collect (e.g., ["dfba_update", "crm_update"])
    """
    config_schema = {"sources": "list[string]"}

    def __init__(self, config, core):
        super().__init__(config, core)
        self.sources = list(self.config["sources"])

    def inputs(self):
        # Each source is a map[set_float] (like dfba_update/crm_update)
        return {src: "map[set_float]" for src in self.sources}

    def outputs(self):
        return {"species_updates": "map[map[set_float]]"}

    def update(self, inputs):
        merged = {}
        for src in self.sources:
            if src not in inputs:
                continue
            upd = inputs[src]
            # upd is a flat map: {species_or_resource_id: delta, ...}
            # We group by species key for the top level.
            # Convention: the species key is present as a key in the same flat map (like dFBA & CRM outputs).
            # We put the whole flat map under that species.
            # If you produce multiple species from one process, make one flat map per species instead.
            # Here: single-species per process => species key is unique.
            species_keys = [k for k in upd.keys() if not k.startswith("EX_")]  # crude heuristic
            if len(species_keys) == 1:
                sp = species_keys[0]
                merged[sp] = upd
            else:
                # fallback: store entire map under the src name
                merged[src] = upd
        return {"species_updates": merged}


# assuming: dFBA, UpdateEnvironment, StaticConcentration, WaveFunction, Injector already registered

def make_crm_process_spec(
    species_name: str,
    resource_order: list[str],
    Km: list[float],
    yields_: list[float],
    vmax: list[float],
    B: list[list[float]],
    maintenance: float,
    dilution: float = 0.0,
    feed: list[float] | None = None,
    resource_loss: list[float] | None = None,
    interval: float = 1.0,
):
    return {
        "_type": "process",
        "address": "local:GoldfordCRM",
        "config": {
            "species_name": species_name,
            "resource_order": resource_order,
            "Km": Km,
            "yields": yields_,
            "vmax": vmax,
            "B": B,
            "maintenance": maintenance,
            "dilution": dilution,
            "feed": feed,
            "resource_loss": resource_loss,
        },
        "inputs": { "shared_environment": ["shared_environment"] },
        "outputs": { "crm_update": ["crm_update"] },
        "interval": interval,
    }

def make_collector_spec(sources: list[str], interval: float = 1.0):
    return {
        "_type": "step",
        "address": "local:CollectSpeciesUpdates",
        "config": {"sources": sources},
        "inputs": {src: [src] for src in sources},
        "outputs": {"species_updates": ["species_updates"]},
        "interval": interval,
    }


def make_update_env_spec(interval: float = 1.0):
    return {
        "_type": "step",
        "address": "local:UpdateEnvironment",
        "inputs": {
            "shared_environment": ["shared_environment"],
            "species_updates": ["species_updates"],
        },
        "outputs": { "counts": ["shared_environment", "counts"] },
        "interval": interval,
    }


# --- Example composite (one dFBA + one GoldfordCRM) ---
def make_hybrid_composite(dfba_spec_for_ecoli, crm_param_block, volume=1.0):
    """
    dfba_spec_for_ecoli: your existing dFBA spec block (as built by make_cdfba_composite or manually).
                         It must output 'dfba_update' and read 'shared_environment'.
    crm_param_block: dict with CRM matrices/vectors for a single species.
                     keys: species_name, resource_order, Km, yields, vmax, B, maintenance, (dilution/feed/resource_loss optional)
    """

    # 1) stores
    state = {
        "global_time": 0.0,
        "shared_environment": {
            "counts": {},   # fill with initial values later or upstream
            "volume": volume,
        },
        # these are “wiring hubs” (don’t need initial data)
        "dfba_update": {},
        "crm_update": {},
        "species_updates": {},
    }

    # 2) processes/steps
    crm_spec = make_crm_process_spec(
        species_name   = crm_param_block["species_name"],
        resource_order = crm_param_block["resource_order"],
        Km             = crm_param_block["Km"],
        yields_        = crm_param_block["yields"],
        vmax           = crm_param_block["vmax"],
        B              = crm_param_block["B"],
        maintenance    = crm_param_block["maintenance"],
        dilution       = crm_param_block.get("dilution", 0.0),
        feed           = crm_param_block.get("feed"),
        resource_loss  = crm_param_block.get("resource_loss"),
        interval       = crm_param_block.get("interval", 1.0),
    )

    # collector merges dfba + crm into species_updates
    collector_spec = make_collector_spec(
        sources=["dfba_update", "crm_update"], interval=1.0
    )

    update_env_spec = make_update_env_spec(interval=1.0)

    # 3) emitter (optional)
    emitter = emitter_from_wires({
        "global_time": ["global_time"],
        "shared_environment": ["shared_environment"],
    })

    # 4) assemble composite spec
    spec = {
        # stores
        "global_time": state["global_time"],
        "shared_environment": state["shared_environment"],

        # add passed dFBA block (must wire: inputs shared_environment, output dfba_update)
        "dFBA": dfba_spec_for_ecoli,

        # add CRM
        "GoldfordCRM": crm_spec,

        # the collector
        "CollectSpeciesUpdates": collector_spec,

        # environment updater
        "UpdateEnvironment": update_env_spec,

        # emit
        "emitter": emitter,
    }
    return spec

