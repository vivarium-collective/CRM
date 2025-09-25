# ========= 0) imports =========
from dataclasses import dataclass
from typing import Dict, List, Optional
from cobra.io import read_sbml_model
from process_bigraph import Process, Composite, ProcessTypes
from process_bigraph.emitter import gather_emitter_results, emitter_from_wires
from cobra.medium import minimal_medium
from cdFBA.utils import get_substrates, get_reaction_map, get_exchanges, model_from_file
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_all_substrates(models, exchanges):
    resource_names = []
    for model in models:
        resource_names.extend(get_substrates(model, exchanges))
    resource_names = list(set(resource_names))
    return resource_names


def get_all_reaction_map(models, exchanges):
    reaction_map = {}
    for model in models:
        reaction_map.update(get_reaction_map(model, exchanges))
    return reaction_map


def get_mini_medium(model, target_growth=10e-4):
    ## make a copy of the model
    try_model = model.copy()

    ## compute the minimum media neccessary for growth at targeted_growth
    mini_growth = minimal_medium(try_model, target_growth, minimize_components=10, open_exchanges=True)

    ## set minimal medium
    mini_medium = {}
    for i in mini_growth.index:
        mini_medium[i] = try_model.medium[i]

    ## get all other metabolites that can be added to the medium
    additional_medium = try_model.medium
    for i in mini_medium.keys():
        del additional_medium[i]
    return mini_medium, additional_medium


def get_reaction_map(model_file="textbook", exchanges=None):
    """Returns a reaction_name_map dictionary from a medium dictionary as obtained
    from model.medium or cobra.medium.minimum_medium()
    Parameters:
        model_file : str, file path or BiGG Model ID
        exchanges : lst, list of names of substrates required by the model organism
    Returns:
        reaction_name_map : dict, maps substrate names to reactions
    """
    if isinstance(model_file, str):
        model = model_from_file(model_file)
    else:
        model = model_file
    if exchanges is None:
        exchanges = get_exchanges(model)
    reaction_map = {
        list(getattr(model.reactions, i).metabolites.keys())[0].name: i
        for i in exchanges if hasattr(model.reactions, i)
    }
    # substrates = get_substrates(model, exchanges)
    # ids = exchanges
    # reaction_name_map = {}
    # for i in range(len(substrates)):
    #     reaction_name_map[substrates[i]] = ids[i]
    return reaction_map


def invert_reaction_map(name_to_exid: dict) -> dict:
    return {ex: name for name, ex in name_to_exid.items()}


def remap_extractions_ids_to_names(extractions_by_species: dict, name_to_exid: dict) -> dict:
    exid_to_name = invert_reaction_map(name_to_exid)
    out = {}
    for sp, packs in extractions_by_species.items():
        out[sp] = {}
        # flat maps
        for key in ("yields", "uptakes"):
            m = packs.get(key, {}) or {}
            out[sp][key] = {exid_to_name[k]: v for k, v in m.items() if k in exid_to_name}
        # nested byproducts
        bp = {}
        for consumed_exid, prod_map in (packs.get("byproducts", {}) or {}).items():
            if consumed_exid not in exid_to_name:
                continue
            consumed_name = exid_to_name[consumed_exid]
            inner = {exid_to_name[k]: v for k, v in (prod_map or {}).items() if k in exid_to_name}
            bp[consumed_name] = inner
        out[sp]["byproducts"] = bp
    return out

# ======== 1) helpers (IDs, biomass, env) ========
def detect_biomass_id(model):
    # Prefer objective if a single variable; else scan for "biomass/growth"
    try:
        return list(model.objective.variables)[0].name
    except Exception:
        pass
    for rxn in model.reactions:
        nm = (rxn.id + " " + rxn.name).lower()
        if "biomass" in nm or "growth" in nm:
            return rxn.id
    raise ValueError("Biomass reaction not found")

def initial_environment(*, volume: float, species_list: List[str], substrates: List[str],
                        biomass_overrides: Optional[Dict[str,float]]=None,
                        resource_overrides: Optional[Dict[str,float]]=None,
                        default_biomass: float=0.1, default_resource: float=0.0):
    biomass_overrides = biomass_overrides or {}
    resource_overrides = resource_overrides or {}
    counts = {}
    # resources (EX IDs)
    for ex in substrates:
        counts[ex] = float(resource_overrides.get(ex, default_resource))
    # species biomasses
    for sp in species_list:
        counts[sp] = float(biomass_overrides.get(sp, default_biomass))
    conc = {k: v/volume for k, v in counts.items()}
    return {"volume": float(volume), "counts": counts, "concentrations": conc}

# ======== 2) your minimal CRM (ODE) for ER ========
@dataclass
class MCRMParams:
    yields: np.ndarray      # (m,p) gDW/mmol
    vmax: np.ndarray        # (m,p) mmol/(gDW·h)
    B: np.ndarray           # (m,p,p) produced_i per consumed_j
    Km: np.ndarray          # (p,)   mmol/L
    maintenance: np.ndarray # (m,)   1/h
    dilution: float = 0.0
    feed: Optional[np.ndarray] = None
    resource_loss: Optional[np.ndarray] = None
    def validate(self):
        m,p = self.yields.shape
        assert self.vmax.shape==(m,p)
        assert self.B.shape==(m,p,p)
        assert self.Km.shape==(p,)
        assert self.maintenance.shape==(m,)
        if self.dilution!=0.0:
            assert self.feed is not None and self.feed.shape==(p,)
        if self.resource_loss is not None:
            assert self.resource_loss.shape==(p,)

def _monod_u(R, Km): return R/(Km+R+1e-12)

def build_params(species_names, resource_names, *,
                 yields_map, vmax_map, maintenance_map, Km_map, byproducts_map,
                 dilution=0.0, feed_map=None, resource_loss_map=None, byproduct_sign="as_is"):
    m,p = len(species_names), len(resource_names)
    s_idx = {s:i for i,s in enumerate(species_names)}
    r_idx = {r:j for j,r in enumerate(resource_names)}
    Y  = np.zeros((m,p)); V = np.zeros((m,p)); B = np.zeros((m,p,p)); Km = np.zeros(p); M = np.zeros(m)
    feed = None
    if dilution>0.0:
        feed = np.zeros(p); feed_map = feed_map or {}
        for r,j in r_idx.items(): feed[j] = float(feed_map.get(r,0.0))
    loss = None
    if resource_loss_map is not None:
        loss = np.zeros(p);
        for r,j in r_idx.items(): loss[j] = float(resource_loss_map.get(r,0.0))
    for s,si in s_idx.items():
        M[si] = float(maintenance_map.get(s,0.02))
        yrow = yields_map.get(s,{}) or {}
        vrow = vmax_map.get(s,{}) or {}
        for r,rj in r_idx.items():
            Y[si,rj] = float(yrow.get(r,0.0))
            V[si,rj] = float(vrow.get(r,0.0))
    for s,si in s_idx.items():
        for consumed, produceds in (byproducts_map.get(s,{}) or {}).items():
            j = r_idx.get(consumed);
            if j is None: continue
            for produced, val in (produceds or {}).items():
                i = r_idx.get(produced);
                if i is None: continue
                raw = float(val)
                stoich = abs(raw) if byproduct_sign=="abs" else (-raw if byproduct_sign=="neg_to_pos" else raw)
                B[si,i,j] = stoich
    for r,rj in r_idx.items(): Km[rj] = float(Km_map.get(r,0.5))
    P = MCRMParams(Y,V,B,Km,M,float(dilution),feed,loss); P.validate(); return P

class MCRM_Process(Process):
    config_schema = {
        "species_names": "list[string]",
        "resource_names": "list[string]",
        "params": "any",
        "method": {"_type":"string","_default":"RK45"},
        "rtol":{"_type":"float","_default":1e-5},
        "atol":{"_type":"float","_default":1e-7},
        "max_step":{"_type":"float","_default":0.05},
        "clip_nonnegative":{"_type":"boolean","_default":True},
    }
    def initialize(self, config):
        self.species_names = list(config["species_names"])
        self.resource_names= list(config["resource_names"])
        self.params = config["params"]; self.params.validate()
        m,p = self.params.yields.shape
        assert len(self.species_names)==m and len(self.resource_names)==p
        self.method = {"rk23":"RK23","rk45":"RK45","dop853":"DOP853","radau":"Radau","bdf":"BDF","lsoda":"LSODA"}\
                      .get(str(config.get("method","RK45")).lower(),"RK45")
        self.rtol=float(config.get("rtol",1e-5)); self.atol=float(config.get("atol",1e-7))
        self.max_step=float(config.get("max_step",0.05)); self.clip=bool(config.get("clip_nonnegative",True))
        self.m,self.p = m,p

    def inputs(self):  return {"shared_environment":"volumetric"}


    def outputs(self): return {"update":"map[float]"}


    def _ode(self,t,y):
        m,p=self.m,self.p; n=y[:m]; R=y[m:]
        P=self.params
        u = _monod_u(np.maximum(R,0.0), P.Km)
        v = P.vmax * u[None,:]
        growth = (P.yields*v).sum(axis=1)
        dn_dt  = n * (growth - P.maintenance - P.dilution)
        cons = (n[:,None]*v).sum(axis=0)
        produced = np.zeros(p)
        for s in range(m): produced += n[s]*(P.B[s]@v[s])
        feed_term = np.zeros(p) if P.dilution==0.0 else P.dilution*(P.feed - R)
        loss_term = np.zeros(p) if P.resource_loss is None else P.resource_loss*R
        dR_dt = feed_term - cons + produced - loss_term
        return np.concatenate([dn_dt,dR_dt])


    def update(self, inputs, interval):
        env = inputs["shared_environment"]
        # read concentrations
        n0 = np.array([env["concentrations"].get(s,0.0) for s in self.species_names],float)
        R0 = np.array([env["concentrations"].get(r,0.0) for r in self.resource_names],float)
        y0 = np.concatenate([n0,R0])
        from scipy.integrate import solve_ivp
        sol=solve_ivp(self._ode, [0.0, float(interval)], y0, method=self.method, rtol=self.rtol, atol=self.atol,
                      max_step=self.max_step,t_eval=[float(interval)])
        if not sol.success: raise RuntimeError(sol.message)
        y1 = sol.y[:, -1];
        if self.clip:
            y1=np.maximum(y1,0.0)
        m, p = self.m, self.p; n1, R1=y1[:m], y1[m:]
        V=float(env["volume"])
        species_delta = {s: float((n1[i]-n0[i])*V) for i, s in enumerate(self.species_names)}
        resource_delta = {r: float((R1[j]-R0[j])*V) for j, r in enumerate(self.resource_names)}
        update = {**species_delta, **resource_delta}
        return {"update": update}


def extract_yields_agora(model, resource_exchange_ids, *, biomass_rxn_id, use_pfba=False):
    yields={}
    for ex_id in resource_exchange_ids:
        with model as m:
            if ex_id not in m.reactions: yields[ex_id]=0.0; continue
            rxn=m.reactions.get_by_id(ex_id); rxn.lower_bound=-1000.0
            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol = m.optimize() if not use_pfba else m.optimize().fluxes  # simple path
            if getattr(sol, "status", "optimal")!="optimal":
                yields[ex_id]=0.0; continue
            v=float(sol.fluxes.get(ex_id,0.0)); uptake=max(0.0,-v)
            mu=float(sol.objective_value or 0.0)
            yields[ex_id]= mu/uptake if uptake>1e-12 and mu>0 else 0.0
    return yields

def extract_byproducts_agora(model, resource_exchange_ids, *, biomass_rxn_id, use_pfba=False):
    out={}
    for ex_id in resource_exchange_ids:
        with model as m:
            if ex_id not in m.reactions: out[ex_id]={}; continue
            m.reactions.get_by_id(ex_id).lower_bound=-1000.0
            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol = m.optimize()
            if sol.status!="optimal": out[ex_id]={}; continue
            secreted={}
            for ex in m.exchanges:
                f=float(sol.fluxes.get(ex.id,0.0))
                if f>1e-9: secreted[ex.id]=f
            out[ex_id]=secreted
    return out

def extract_uptake_rates_agora(model, resource_exchange_ids, *, biomass_rxn_id, use_pfba=False):
    upt={}
    for ex_id in resource_exchange_ids:
        with model as m:
            if ex_id not in m.reactions: upt[ex_id]=0.0; continue
            m.reactions.get_by_id(ex_id).lower_bound=-1000.0
            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol=m.optimize()
            if sol.status!="optimal": upt[ex_id]=0.0; continue
            v=float(sol.fluxes.get(ex_id,0.0)); upt[ex_id]=max(0.0,-v)
    return upt



def plot_species_and_resources(results,
                               species_keys=None,
                               resource_keys=None,
                               logy=True,
                               figsize=(8, 8),
                               title="Hybrid cdFBA + CRM dynamics",
                               savepath="crm_dfba_dynamics.png",
                               display=True):
    """
    Plot species (bottom) and resources (top) from emitter `results` in two subplots.

    Parameters
    ----------
    results : list[dict]
        Output from gather_emitter_results(sim)[("emitter",)].
    species_keys : list[str] | None
        Names treated as species. If None, detect heuristically.
    resource_keys : list[str] | None
        Names treated as resources. If None, use non-species keys.
    logy : bool
        Use log scale on y-axes (helpful for big dynamic ranges).
    figsize : tuple
        Figure size.
    title : str
        Figure title.
    savepath : str | None
        If provided, saves the figure here (PNG). Set to None to skip saving.
    display : bool
        If True, attempts to show() unless backend is non-interactive.
    """
    # --- extract time & env snapshots ---
    T = [snap["global_time"] for snap in results]
    envs = [snap["shared_environment"]["concentrations"] for snap in results]

    # Collect all keys
    all_keys = sorted({k for e in envs for k in e.keys()})

    # Heuristic species detection if none provided
    if species_keys is None:
        species_keys = [k for k in all_keys if k and k[0].isupper()]
    if resource_keys is None:
        resource_keys = [k for k in all_keys if k not in species_keys]

    def series_for(keys):
        return {k: np.array([float(env.get(k, np.nan)) for env in envs]) for k in keys}

    # Build figure
    fig, (ax_res, ax_spc) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Resources (top)
    for k, y in series_for(resource_keys).items():
        ax_res.plot(T, y, label=k)
    ax_res.set_ylabel("Concentration", fontsize=15)
    if logy: ax_res.set_yscale("log")
    ax_res.set_title("Resources", fontsize=15)
    ax_res.legend(fontsize=15, frameon=False)
    ax_res.grid(alpha=0.3)

    # Species (bottom)
    for k, y in series_for(species_keys).items():
        ax_spc.plot(T, y, label=k)
    ax_spc.set_xlabel("Time (h)", fontsize=15)
    ax_spc.set_ylabel("Biomass", fontsize=15)
    if logy: ax_spc.set_yscale("log")
    ax_spc.set_title("Species", fontsize=15)
    ax_spc.legend(fontsize=15, frameon=False)
    ax_spc.grid(alpha=0.3)

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save safely (works in any backend)
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    # Only show on interactive backends
    backend = matplotlib.get_backend().lower()
    if display and not any(b in backend for b in ("agg", "template")):
        plt.show()

    plt.close(fig)
    return savepath

# ======== 5) Build-and-run (minimal hybrid: BT via dFBA, ER via CRM) ========
if __name__=="__main__":
    # --- models ---
    gut_models = {
        "B_thetaiotaomicron": "/Users/edwin/Downloads/reconstructions/sbml/Bacteroides_thetaiotaomicron_VPI_5482.xml",
        "E_rectale": "/Users/edwin/Downloads/reconstructions/GSMM/Eubacterium_rectale_ATCC_33656.xml",
        "Methanobrevibacter_smithii": "/Users/edwin/Downloads/reconstructions/sbml/Methanobrevibacter_smithii_ATCC_35061.xml",
    }
    models = {k: read_sbml_model(v) for k, v in gut_models.items()}

    BT = models["B_thetaiotaomicron"]
    ER = models["E_rectale"]

    # minimal media just to avoid infeasible models during extraction
    BT_mini_medium = get_mini_medium(BT)[0]
    BT_mini_medium["EX_MGlcn175_rl(e)"] = 0
    ER_mini_medium = get_mini_medium(ER)[0]
    ER_mini_medium["EX_ac(e)"] = 1000  # allow ER to access acetate for extraction
    BT.medium = BT_mini_medium
    ER.medium = ER_mini_medium

    # --- resources ---
    resources_exids = ["EX_hspg(e)", "EX_ac(e)", "EX_but(e)"]

    # name<->id maps (for name-keyed CRM & env)
    reaction_map = get_all_reaction_map([BT, ER], resources_exids)
    resources_names = list(reaction_map.keys())  # ['heparan sulfate proteoglycan','acetate','butyrate']
    resources_exids = list(reaction_map.values())  # ['EX_hspg(e)','EX_ac(e)','EX_but(e)']
    print(resources_names)
    print(reaction_map)

    # --- biomass rxns
    biomass = {"B_thetaiotaomicron": "EX_biomass(e)", "E_rectale": "EX_biomass(e)"}

    # --- GSMM extraction using EX IDs ---
    extractions = {
        "B_thetaiotaomicron": {
            "yields": extract_yields_agora(BT, resources_exids, biomass_rxn_id=biomass["B_thetaiotaomicron"]),
            "byproducts": extract_byproducts_agora(BT, resources_exids, biomass_rxn_id=biomass["B_thetaiotaomicron"]),
            "uptakes": extract_uptake_rates_agora(BT, resources_exids, biomass_rxn_id=biomass["B_thetaiotaomicron"]),
        },
        "E_rectale": {
            "yields": extract_yields_agora(ER, resources_exids, biomass_rxn_id=biomass["E_rectale"]),
            "byproducts": extract_byproducts_agora(ER, resources_exids, biomass_rxn_id=biomass["E_rectale"]),
            "uptakes": extract_uptake_rates_agora(ER, resources_exids, biomass_rxn_id=biomass["E_rectale"]),
        },
    }

    # --- remap extractor outputs to NAME-keyed dicts ---
    extractions_names = remap_extractions_ids_to_names(extractions, reaction_map)
    print(extractions_names)

    # --- CRM (ER) parameter maps (NAME-keyed) ---
    yields_map = {"E_rectale": (extractions_names["E_rectale"]["yields"] or {})}
    vmax_map = {"E_rectale": (extractions_names["E_rectale"]["uptakes"] or {})}
    # Force ER to convert acetate → butyrate with fixed stoichiometry
    byproducts_map = {
        "E_rectale": {
            "acetate": {"butyrate": 0.7}
        }
    }
    Km_map = {"heparan sulfate proteoglycan": 0.5, "acetate": 0.3, "butyrate": 0.2}
    maintenance = {"E_rectale": 0.02}

    # --- pack CRM params (ER-only, NAME-keyed resources) ---
    params = build_params(
        species_names=["E_rectale"],
        resource_names=resources_names,
        yields_map=yields_map,
        vmax_map=vmax_map,
        maintenance_map=maintenance,
        Km_map=Km_map,
        byproducts_map=byproducts_map,
        dilution=0.0,
    )

    # --- environment (NAME-keyed) ---
    env = initial_environment(
        volume=1.0,
        species_list=["B_thetaiotaomicron", "E_rectale"],
        substrates=resources_names,
        biomass_overrides={"B_thetaiotaomicron": 0.1, "E_rectale": 0.1},
        resource_overrides={"heparan sulfate proteoglycan": 20, "acetate": 0.0, "butyrate": 0.0},
        default_resource=0.0,
    )

    # ===== Compose processes =====
    from cdFBA import register_types
    from cdFBA.processes.dfba import dFBA, UpdateEnvironment

    core = ProcessTypes()
    core = register_types(core)
    core.register_process("MCRM_Process", MCRM_Process)
    core.register_process("dFBA", dFBA)
    core.register_process("UpdateEnvironment", UpdateEnvironment)

    # IMPORTANT: give BT only the substrate it consumes (avoid KeyError)
    bt_reaction_map = {
        "heparan sulfate proteoglycan": reaction_map["heparan sulfate proteoglycan"],
        "acetate": reaction_map["acetate"]

    }

    spec = {
        # ER via CRM (NAME-keyed)
        "ER_CRM": {
            "_type": "process", "address": "local:MCRM_Process",
            "config": {
                "species_names": ["E_rectale"],
                "resource_names": resources_names,
                "params": params,
                "method": "RK45", "rtol": 1e-5, "atol": 1e-7, "max_step": 0.05,
                "clip_nonnegative": True,
            },
            "inputs": {"shared_environment": ["Shared Environment"]},
            "outputs": {"update": ["dFBA Results", "ER"]},
            "interval": 0.1,
        },

        # BT via dFBA (NAME-keyed I/O; names→EX IDs via bt_reaction_map)
        "BT_dFBA": {
            "_type": "process", "address": "local:dFBA",
            "config": {
                "model_file": "/Users/edwin/Downloads/BT_min_med_hspg1000.xml",
                "name": "B_thetaiotaomicron",
                "kinetics": {"heparan sulfate proteoglycan": (0.5, 20),
                             "acetate": (0.5, 2)},
                "reaction_map": bt_reaction_map,  # reduced map
                "bounds": {},
                "changes": {"gene_knockout": [], "reaction_knockout": [], "bounds": {}, "kinetics": {}},
            },
            "inputs": {"shared_environment": ["Shared Environment"], "current_update": ["dFBA Results"]},
            "outputs": {"dfba_update": ["dFBA Results", "BT"]},
            "interval": 0.1,
        },

        # shared env + merger
        "Shared Environment": env,
        "UpdateEnv": {
            "_type": "process", "address": "local:UpdateEnvironment", "config": {},
            "inputs": {"shared_environment": ["Shared Environment"], "species_updates": ["dFBA Results"]},
            "outputs": {"counts": ["Shared Environment", "counts"]},
        },

        # wiring & emitter
        "dFBA Results": {"BT": {}, "ER": {}},
        "emitter": emitter_from_wires({
            "global_time": ["global_time"],
            "shared_environment": ["Shared Environment"],
            "dfba_results": ["dFBA Results"]
        }),
    }

    # seed zeros using NAMES + species (not EX IDs!)
    all_keys = resources_names + ["B_thetaiotaomicron", "E_rectale"]
    spec["dFBA Results"]["BT"] = {k: 0.0 for k in all_keys}
    spec["dFBA Results"]["ER"] = {k: 0.0 for k in all_keys}

    # run
    sim = Composite({"state": spec}, core=core)
    sim.run(2)
    results = gather_emitter_results(sim)[("emitter",)]

    plot_species_and_resources(
        results,
        species_keys=["B_thetaiotaomicron", "E_rectale"],
        resource_keys=["heparan sulfate proteoglycan", "acetate", "butyrate"],
        logy=False,
        savepath="hybrid_dynamics_0.7.png",  # file will be created in your cwd
        display=False  # skip show() to avoid PyCharm backend issues
    )
