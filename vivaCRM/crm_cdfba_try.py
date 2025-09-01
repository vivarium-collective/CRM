# --- imports & your existing pieces ---
from process_bigraph import Process, Step, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

# (Use your existing dFBA + UpdateEnvironment classes as defined in your message)
# from your_module import dFBA, UpdateEnvironment

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------
# 1) Goldford-style CRM step
# ---------------------------
def _monod(R: np.ndarray, Km: np.ndarray) -> np.ndarray:
    return R / (Km + R + 1e-12)

class GoldfordCRM(Process):
    """
    Single-species Goldford MCRM over one interval, reading/writing shared_environment.

    Config:
      species_name: str
      resource_order: list[str]           # resource IDs (must be keys in shared_environment["counts"])
      Km: list[float]                     # length p
      yields: list[float]                 # length p (gDW/mmol)
      vmax: list[float]                   # length p (mmol/(gDW·h))
      B: list[list[float]]                # (p,p) produced_i per consumed_j
      maintenance: float                  # 1/h
      dilution: float = 0.0               # 1/h (chemostat)
      feed: optional list[float]          # length p (if dilution>0)
      resource_loss: optional list[float] # length p (extra loss)
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
        self.sp = config["species_name"]
        self.res_ids = list(config["resource_order"])
        p = len(self.res_ids)
        self.Km = np.array(config["Km"], float)
        self.Y  = np.array(config["yields"], float).reshape(1, p)
        self.V  = np.array(config["vmax"],   float).reshape(1, p)
        self.B  = np.array(config["B"],      float).reshape(1, p, p)
        self.m  = np.array([config["maintenance"]], float)
        self.delta = float(config.get("dilution", 0.0))
        self.feed = None if config.get("feed") is None else np.array(config["feed"], float)
        self.mu_loss = None if config.get("resource_loss") is None else np.array(config["resource_loss"], float)

    def inputs(self):
        return { "shared_environment": "volumetric" }

    def outputs(self):
        # same “flat delta map” shape as dFBA’s dfba_update
        return { "crm_update": "map[set_float]" }

    def _rhs(self, t, x, P):
        n, R = x[0], x[1:]
        u = _monod(R, P["Km"])
        v = P["V"][0] * u
        growth = (P["Y"][0] * v).sum()
        dn = n * (growth - P["m"][0] - P["delta"])
        consumption = n * v
        produced = (P["B"][0] @ v) * n
        feed_term = 0.0*R if P["delta"]==0.0 else P["delta"]*(P["feed"] - R)
        loss_term = 0.0*R if (P["mu_loss"] is None) else P["mu_loss"]*R
        dR = feed_term - consumption + produced - loss_term
        return np.concatenate([[dn], dR])

    def update(self, inputs, interval):
        env = inputs["shared_environment"]
        counts = env["counts"]; Vsys = env["volume"]
        n0 = float(counts[self.sp])
        R0_counts = np.array([counts[r] for r in self.res_ids], float)
        R0 = R0_counts / max(Vsys, 1e-12)

        P = {
            "Km": self.Km, "Y": self.Y, "V": self.V, "B": self.B,
            "m": self.m, "delta": self.delta, "feed": self.feed, "mu_loss": self.mu_loss
        }
        x0 = np.concatenate([[n0], R0])
        sol = solve_ivp(lambda t,x: self._rhs(t,x,P), (0.0, float(interval)), x0,
                        t_eval=[float(interval)], method="RK45", rtol=1e-7, atol=1e-9)
        xF = np.maximum(sol.y[:, -1], 0.0)
        nF, RF = xF[0], xF[1:]

        update = {}
        update[self.sp] = nF - n0
        RF_counts = RF * Vsys
        dR_counts = RF_counts - R0_counts
        for rid, dval in zip(self.res_ids, dR_counts):
            update[rid] = float(dval)
        return {"crm_update": update}

# ------------------------------------
# 2) Merge {dfba_update, crm_update} → species_updates
# ------------------------------------
class CollectSpeciesUpdates(Step):
    config_schema = {"sources": "list[string]"}
    def __init__(self, config, core):
        super().__init__(config, core)
        self.sources = list(self.config["sources"])
    def inputs(self):
        return {src: "map[set_float]" for src in self.sources}
    def outputs(self):
        return {"species_updates": "map[map[set_float]]"}
    def update(self, inputs):
        merged = {}
        for src in self.sources:
            if src not in inputs: continue
            upd = inputs[src]
            # Heuristic: the species key is the one that is not an EX_ resource
            species_keys = [k for k in upd.keys() if not k.startswith("EX_")]
            if len(species_keys) == 1:
                merged[species_keys[0]] = upd
            else:
                # fallback: store by source name
                merged[src] = upd
        return {"species_updates": merged}

# ------------------------------------------------
# 3) Helper: build process specs to wire into Composite
# ------------------------------------------------
def make_dfba_process_spec(*, name, model_file, reaction_map, kinetics, bounds=None, interval=1.0):
    return {
        "_type": "process",
        "address": "local:dFBA",
        "config": {
            "model_file": model_file,
            "name": name,
            "reaction_map": reaction_map,  # e.g. {"EX_glc__D_e":"EX_glc__D_e","EX_ac_e":"EX_ac_e"}
            "kinetics": kinetics,          # e.g. {"EX_glc__D_e": (Km,Vmax), ...}
            "bounds": bounds or {"EX_o2_e": {"lower": -2, "upper": None}, "ATPM": {"lower": 1, "upper": 1}},
            "changes": {"gene_knockout": [], "reaction_knockout": [], "bounds": {}, "kinetics": {}}
        },
        "inputs": { "shared_environment": ["shared_environment"], "current_update": ["species_updates"] },
        "outputs": { "dfba_update": ["dfba_update"] },
        "interval": interval,
    }

def make_crm_process_spec(crm_params, interval=1.0):
    return {
        "_type": "process",
        "address": "local:GoldfordCRM",
        "config": crm_params,  # dict with keys defined in GoldfordCRM.config_schema
        "inputs":  { "shared_environment": ["shared_environment"] },
        "outputs": { "crm_update": ["crm_update"] },
        "interval": interval,
    }

def make_collector_spec(sources, interval=1.0):
    return {
        "_type": "step",
        "address": "local:CollectSpeciesUpdates",
        "config": {"sources": sources},
        "inputs": {src: [src] for src in sources},
        "outputs": {"species_updates": ["species_updates"]},
        "interval": interval,
    }

def make_update_env_spec(interval=1.0):
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

if __name__ == "__main__":
    # ------------------------------------
    # 4) Build a runnable hybrid composite
    # ------------------------------------
    def run_hybrid_demo():
        # (A) Register types & processes
        from cdFBA.process import cdFBA
        core = ProcessTypes()
        core = register_types(core)
        core.register_process("dFBA", dFBA)
        core.register_process("UpdateEnvironment", UpdateEnvironment)
        core.register_process("GoldfordCRM", GoldfordCRM)
        core.register_step("CollectSpeciesUpdates", CollectSpeciesUpdates)

        # (B) Pick shared resources and keep naming consistent across BOTH processes
        resources = ["EX_glc__D_e", "EX_ac_e"]  # must match exchange IDs & env keys

        # (C) dFBA process (E. coli)
        dfba_spec = make_dfba_process_spec(
            name="E.coli",
            model_file="iAF1260",  # BiGG id (or path)
            reaction_map={r: r for r in resources},  # use EX_* names as env keys
            kinetics={"EX_glc__D_e": (0.02, 15), "EX_ac_e": (0.5, 7)},  # Km, Vmax
            bounds={"EX_o2_e": {"lower": -2, "upper": None}, "ATPM": {"lower": 1, "upper": 1}},
            interval=1.0
        )

        # (D) CRM process params (single-species “CRM_Bug” on same resources)
        crm_params = {
            "species_name": "CRM_Bug",
            "resource_order": resources,
            "Km": [0.5, 0.5],
            "yields": [0.05, 0.03],  # gDW/mmol
            "vmax": [12.0, 8.0],  # mmol/(gDW·h)
            "B": [[0.0, 0.0],  # byproduct matrix (glc->acetate overflow)
                  [0.5, 0.0]],
            "maintenance": 0.02,
            "dilution": 0.0,
            "feed": None,
            "resource_loss": None,
        }
        crm_spec = make_crm_process_spec(crm_params, interval=1.0)

        # (E) Collector & updater
        collector_spec = make_collector_spec(["dfba_update", "crm_update"], interval=1.0)
        update_env_spec = make_update_env_spec(interval=1.0)

        # (F) Emitter (optional)
        emitter = emitter_from_wires({
            "global_time": ["global_time"],
            "shared_environment": ["shared_environment"],
        })

        # (G) Initial state (counts; resources as counts = concentration * volume)
        volume = 1.0
        initial_counts = {
            # species biomasses
            "E.coli": 0.1,
            "CRM_Bug": 0.05,
            # resources
            "EX_glc__D_e": 40.0 * volume,
            "EX_ac_e": 0.0 * volume,
        }

        spec = {
            # stores
            "global_time": 0.0,
            "shared_environment": {"counts": initial_counts, "volume": volume},
            # “hub” stores (no init data needed, but present in wiring)
            "dfba_update": {},
            "crm_update": {},
            "species_updates": {},
            # processes/steps
            "dFBA": dfba_spec,
            "GoldfordCRM": crm_spec,
            "CollectSpeciesUpdates": collector_spec,
            "UpdateEnvironment": update_env_spec,
            "emitter": emitter,
        }

        # (H) Run
        sim = Composite({"state": spec}, core=core)
        sim.run(20)
        results = gather_emitter_results(sim)[("emitter",)]

        # (I) Quick plots
        times = [r["global_time"] for r in results]
        counts = [r["shared_environment"]["counts"] for r in results]

        def series(key): return [c.get(key, 0.0) for c in counts]

        plt.figure()
        plt.plot(times, series("E.coli"), label="E.coli (dFBA)")
        plt.plot(times, series("CRM_Bug"), label="CRM_Bug (Goldford)")
        plt.xlabel("time");
        plt.ylabel("biomass");
        plt.legend();
        plt.title("Hybrid populations")

        plt.figure()
        plt.plot(times, series("EX_glc__D_e"), label="Glucose")
        plt.plot(times, series("EX_ac_e"), label="Acetate")
        plt.xlabel("time");
        plt.ylabel("counts");
        plt.legend();
        plt.title("Shared resources")
        plt.show()


    # Fire it up
    run_hybrid_demo()

