"""
GSMM → MCRM Utilities (Goldford-style)
--------------------------------------
Utilities to:
  - extract yields, byproducts, uptakes from SBML/COBRA models,
  - build CRM parameter matrices,
  - simulate the Goldford-style MCRM,
  - plot results.

Deps: cobra, numpy, scipy, matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.integrate import solve_ivp

# COBRA is optional until you call extract_* functions
try:
    import cobra
    from cobra.flux_analysis import pfba as cobra_pfba
except Exception:
    cobra = None
    cobra_pfba = None

import numpy as np

# ============================================================================
# 0) Helpers for a reasonable medium during extraction
# ============================================================================

def _open_base_medium(m, o2_lb: Optional[float] = None) -> None:
    """
    Opens a simple inorganic base medium; optionally caps oxygen uptake.
    Uptake is negative in COBRA (LB < 0).
    """
    # Close all exchanges, then open base
    for ex in m.exchanges:
        ex.lower_bound = 0.0
        ex.upper_bound = 1000.0
    base_open = [
        "EX_h_e", "EX_h2o_e", "EX_na1_e", "EX_k_e", "EX_pi_e",
        "EX_so4_e", "EX_nh4_e", "EX_cl_e", "EX_mg2_e", "EX_ca2_e",
    ]
    for ex_id in base_open:
        if ex_id in m.reactions:
            r = m.reactions.get_by_id(ex_id)
            r.lower_bound = -1000.0
            r.upper_bound = 1000.0
    # Oxygen
    if o2_lb is not None:
        for cand in ("EX_o2_e", "o2_e", "EX_o2(s)_e"):
            if cand in m.reactions:
                r = m.reactions.get_by_id(cand)
                r.lower_bound = -abs(o2_lb)
                r.upper_bound = 1000.0
                break


def _pin_atpm(m, atpm_flux: Optional[float]) -> None:
    """Optionally fix ATP maintenance (ATPM) to a given flux."""
    if atpm_flux is None:
        return
    try:
        rxn = m.reactions.get_by_id("ATPM")
        rxn.lower_bound = float(atpm_flux)
        rxn.upper_bound = float(atpm_flux)
    except Exception:
        # Some models may use different IDs; silently skip
        pass


def _solve(m, use_pfba: bool):
    """Solve FBA or pFBA returning a solution-like object with .fluxes and .objective_value."""
    if use_pfba and cobra_pfba is not None:
        return cobra_pfba(m)
    return m.optimize()


# ============================================================================
# 1) EXTRACTORS: yields, byproducts, uptake rates
# ============================================================================

def extract_yields(
    model,
    resource_exchange_ids: List[str],
    *,
    biomass_rxn_id: str,
    uptake_cap: float = 10.0,
    o2_lb: Optional[float] = None,
    atpm_flux: Optional[float] = None,
    use_pfba: bool = False,
) -> Dict[str, float]:
    """
    Average growth yield per resource (biomass per mmol substrate) from single-resource FBAs.
    Y_i = mu / v_i(actual), using actual uptake (not just the cap).
    """
    if cobra is None:
        raise RuntimeError("COBRApy not available.")
    yields: Dict[str, float] = {}
    for ex_id in resource_exchange_ids:
        with model as m:
            _open_base_medium(m, o2_lb=o2_lb)
            _pin_atpm(m, atpm_flux)
            # Allow only this resource to be consumed
            if ex_id not in m.reactions:
                yields[ex_id] = 0.0
                continue
            m.reactions.get_by_id(ex_id).lower_bound = -abs(uptake_cap)

            # Objective: biomass
            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol = _solve(m, use_pfba)
            if sol.status != "optimal":
                yields[ex_id] = 0.0
                continue

            v = float(sol.fluxes.get(ex_id, 0.0))
            v_upt = max(0.0, -v)  # uptake is negative
            mu = float(sol.objective_value or 0.0)
            yields[ex_id] = (mu / v_upt) if v_upt > 1e-12 else 0.0
    return yields


def extract_byproducts(
    model,
    resource_exchange_ids: List[str],
    *,
    biomass_rxn_id: str,
    uptake_cap: float = 10.0,
    o2_lb: Optional[float] = None,
    atpm_flux: Optional[float] = None,
    use_pfba: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    For each consumed resource ex_j, record secreted exchange fluxes {ex_i: flux_i (>=0)}.
    NOTE: Raw secretion fluxes are returned; they’re normalized in the builder to mmol/mmol.
    """
    if cobra is None:
        raise RuntimeError("COBRApy not available.")
    byp: Dict[str, Dict[str, float]] = {}
    for ex_id in resource_exchange_ids:
        with model as m:
            _open_base_medium(m, o2_lb=o2_lb)
            _pin_atpm(m, atpm_flux)
            if ex_id not in m.reactions:
                byp[ex_id] = {}
                continue
            m.reactions.get_by_id(ex_id).lower_bound = -abs(uptake_cap)
            m.objective = m.reactions.get_by_id(biomass_rxn_id)

            sol = _solve(m, use_pfba)
            if sol.status != "optimal":
                byp[ex_id] = {}
                continue

            secreted: Dict[str, float] = {}
            for ex in m.exchanges:
                f = float(sol.fluxes.get(ex.id, 0.0))
                if f > 1e-9:  # secretion
                    secreted[ex.id] = f
            byp[ex_id] = secreted
    return byp


def extract_uptake_rates(
    model,
    resource_exchange_ids: List[str],
    *,
    biomass_rxn_id: str,
    uptake_cap: float = 10.0,
    o2_lb: Optional[float] = None,
    atpm_flux: Optional[float] = None,
    use_pfba: bool = False,
) -> Dict[str, float]:
    """
    Actual single-resource uptake rate (mmol/gDW/h) under the same regime used for yields.
    """
    if cobra is None:
        raise RuntimeError("COBRApy not available.")
    uptakes: Dict[str, float] = {}
    for ex_id in resource_exchange_ids:
        with model as m:
            _open_base_medium(m, o2_lb=o2_lb)
            _pin_atpm(m, atpm_flux)
            if ex_id not in m.reactions:
                uptakes[ex_id] = 0.0
                continue
            m.reactions.get_by_id(ex_id).lower_bound = -abs(uptake_cap)
            m.objective = m.reactions.get_by_id(biomass_rxn_id)

            sol = _solve(m, use_pfba)
            if sol.status != "optimal":
                uptakes[ex_id] = 0.0
                continue

            v = float(sol.fluxes.get(ex_id, 0.0))
            uptakes[ex_id] = max(0.0, -v)
    return uptakes


# ============================================================================
# 2) Parameter container + ODE for the Goldford-style MCRM
# ============================================================================

@dataclass
class MCRMParams:
    """
    Goldford-style MCRM parameters:
      yields: (m,p)   gDW/mmol
      vmax:   (m,p)   mmol/(gDW·h)  (Monod Vmax per species-resource)
      B:      (m,p,p) byproduct stoichiometry (produced_i per consumed_j)
      Km:     (p,)    Monod Km shared across species (mmol/L)
      maintenance: (m,) 1/h
      dilution: float  (chemostat; 0 for batch)
      feed: (p,) or None  (resource feed if dilution>0)
      resource_loss: (p,) or None  (extra loss terms)
    """
    yields: np.ndarray
    vmax: np.ndarray
    B: np.ndarray
    Km: np.ndarray
    maintenance: np.ndarray
    dilution: float = 0.0
    feed: Optional[np.ndarray] = None
    resource_loss: Optional[np.ndarray] = None

    def validate(self) -> None:
        m, p = self.yields.shape
        assert self.vmax.shape == (m, p)
        assert self.B.shape == (m, p, p)
        assert self.Km.shape == (p,)
        assert self.maintenance.shape == (m,)
        if self.dilution != 0.0:
            assert self.feed is not None and self.feed.shape == (p,)


def _monod_u(R: np.ndarray, Km: np.ndarray) -> np.ndarray:
    return R / (Km + R + 1e-12)


def mcrm_rhs(t: float, x: np.ndarray, params: MCRMParams) -> np.ndarray:
    """
    State x = [n_0..n_{m-1}, R_0..R_{p-1}].
    dn_sigma/dt = n_sigma (sum_i Y[sigma,i] * v[sigma,i] - m_sigma - delta)
    dR_i/dt     = delta*(K_i - R_i) - sum_sigma n_sigma v[sigma,i]
                  + sum_sigma n_sigma sum_j B[sigma,i,j] v[sigma,j] - mu_i R_i
    v[sigma,i]  = Vmax[sigma,i] * R_i/(Km_i + R_i)
    """
    params.validate()
    m, p = params.yields.shape
    n = x[:m]
    R = x[m:]

    u = _monod_u(R, params.Km)             # (p,)
    v = params.vmax * u[None, :]           # (m,p)
    growth = (params.yields * v).sum(axis=1)
    dn = n * (growth - params.maintenance - params.dilution)

    consumption = (n[:, None] * v).sum(axis=0)   # (p,)
    produced = np.zeros(p)
    for s in range(m):
        produced += n[s] * (params.B[s] @ v[s])

    feed_term = np.zeros(p) if params.dilution == 0.0 else params.dilution * (params.feed - R)
    loss_term = np.zeros(p) if params.resource_loss is None else params.resource_loss * R
    dR = feed_term - consumption + produced - loss_term

    return np.concatenate([dn, dR])


def simulate_mcrm(
    params: MCRMParams,
    initial_biomass: np.ndarray,   # (m,)
    initial_resources: np.ndarray, # (p,)
    *,
    T: float = 24.0,
    steps: int = 1200,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str = "RK45",
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the MCRM; returns (t, X) with columns [n..., R...]."""
    params.validate()
    x0 = np.concatenate([initial_biomass.astype(float), initial_resources.astype(float)])
    t_eval = np.linspace(0.0, T, steps)
    sol = solve_ivp(lambda t, x: mcrm_rhs(t, x, params),
                    (t_eval[0], t_eval[-1]), x0, t_eval=t_eval,
                    method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T


# ============================================================================
# 3) BUILD PARAMETERS from extractor outputs
# ============================================================================

def build_crm_params_from_extractions(
    extractions: Dict[str, Dict[str, Dict[str, float]]],
    resource_id_order: List[str],
    Km_by_resource_id: Dict[str, float],
    *,
    use_uptake_as_vmax: bool = True,
    vmax_by_species_resource: Optional[Dict[Tuple[str, str], float]] = None,
    maintenance_by_species: Optional[Dict[str, float]] = None,
    dilution: float = 0.0,
    feed_by_resource_id: Optional[Dict[str, float]] = None,
    resource_loss_by_resource_id: Optional[Dict[str, float]] = None,
) -> Tuple[MCRMParams, List[str], List[str]]:
    """
    Convert extractor dict (per species: yields, byproducts, uptakes) into MCRM matrices.

    extractions:
        {
          species: {
            "yields":     {ex_id: Y_biomass/mmol},
            "byproducts": {ex_consumed: {ex_produced: secretion_flux}},
            "uptakes":    {ex_id: uptake_flux},
          }, ...
        }
    """
    species_order = list(extractions.keys())
    m = len(species_order)
    p = len(resource_id_order)

    Y = np.zeros((m, p))
    V = np.zeros((m, p))
    B = np.zeros((m, p, p))
    Km = np.array([Km_by_resource_id[r] for r in resource_id_order], float)
    maint = np.zeros(m)

    for s_idx, sp in enumerate(species_order):
        block = extractions[sp]
        maint[s_idx] = float(maintenance_by_species.get(sp, 0.02)) if maintenance_by_species else 0.02

        # yields & vmax
        for j, ex in enumerate(resource_id_order):
            Y[s_idx, j] = float(block.get("yields", {}).get(ex, 0.0))
            if use_uptake_as_vmax:
                V[s_idx, j] = float(block.get("uptakes", {}).get(ex, 0.0))
            else:
                if vmax_by_species_resource is None:
                    raise ValueError("Provide vmax_by_species_resource when use_uptake_as_vmax=False")
                V[s_idx, j] = float(vmax_by_species_resource.get((sp, ex), 0.0))

        # byproducts normalized to mmol/mmol (produced per consumed)
        upt = block.get("uptakes", {})
        for consumed_ex, produced_map in block.get("byproducts", {}).items():
            if consumed_ex not in resource_id_order:
                continue
            j = resource_id_order.index(consumed_ex)
            v_cons = float(upt.get(consumed_ex, 0.0))
            if v_cons <= 1e-12:
                continue
            for produced_ex, secretion_flux in produced_map.items():
                if produced_ex not in resource_id_order:
                    continue
                i = resource_id_order.index(produced_ex)
                B[s_idx, i, j] = float(secretion_flux) / v_cons

    feed = None if feed_by_resource_id is None else np.array([feed_by_resource_id.get(r, 0.0) for r in resource_id_order], float)
    extra_loss = None if resource_loss_by_resource_id is None else np.array([resource_loss_by_resource_id.get(r, 0.0) for r in resource_id_order], float)

    params = MCRMParams(
        yields=Y, vmax=V, B=B, Km=Km, maintenance=maint,
        dilution=float(dilution), feed=feed, resource_loss=extra_loss,
    )
    return params, species_order, resource_id_order


# ============================================================================
# 4) Plotting helper
# ============================================================================

def plot_mcrm(t: np.ndarray, X: np.ndarray, species_order: List[str], resource_order: List[str]) -> None:
    """
    Quick diagnostic plots. (Matplotlib only; no seaborn.)
    """
    import matplotlib.pyplot as plt
    m = len(species_order)
    p = len(resource_order)

    # Biomass
    plt.figure()
    for i, sp in enumerate(species_order):
        plt.plot(t, X[:, i], label=f"Biomass: {sp}")
    plt.xlabel("Time [h]"); plt.ylabel("Biomass [gDW/L]"); plt.legend()

    # Resources
    plt.figure()
    for j, ex in enumerate(resource_order):
        plt.plot(t, X[:, m + j], label=f"Resource: {ex}")
    plt.xlabel("Time [h]"); plt.ylabel("Concentration [mmol/L]"); plt.legend()
    plt.show()


# ============================================================================
# 5) End-to-end convenience wrapper (optional)
# ============================================================================

def run_mcrm_from_extractions(
    extractions: Dict[str, Dict[str, Dict[str, float]]],
    resource_id_order: List[str],
    Km_by_resource_id: Dict[str, float],
    initial_biomass_by_species: Dict[str, float],
    initial_resources_by_id: Dict[str, float],
    *,
    use_uptake_as_vmax: bool = True,
    vmax_by_species_resource: Optional[Dict[Tuple[str, str], float]] = None,
    maintenance_by_species: Optional[Dict[str, float]] = None,
    dilution: float = 0.0,
    feed_by_resource_id: Optional[Dict[str, float]] = None,
    resource_loss_by_resource_id: Optional[Dict[str, float]] = None,
    T: float = 24.0,
    steps: int = 1200,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], MCRMParams]:
    """
    One-call:
      1) build params from extractor dict,
      2) simulate,
      3) return (t, X, species_order, resource_order, params).
    """
    params, species_order, resource_order = build_crm_params_from_extractions(
        extractions=extractions,
        resource_id_order=resource_id_order,
        Km_by_resource_id=Km_by_resource_id,
        use_uptake_as_vmax=use_uptake_as_vmax,
        vmax_by_species_resource=vmax_by_species_resource,
        maintenance_by_species=maintenance_by_species,
        dilution=dilution,
        feed_by_resource_id=feed_by_resource_id,
        resource_loss_by_resource_id=resource_loss_by_resource_id,
    )

    n0 = np.array([initial_biomass_by_species.get(sp, 0.0) for sp in species_order], float)
    R0 = np.array([initial_resources_by_id.get(ex, 0.0) for ex in resource_order], float)

    t, X = simulate_mcrm(params, n0, R0, T=T, steps=steps)
    return t, X, species_order, resource_order, params



# ============================================================================
# 6) Implementation in Vivarium
# ============================================================================

def build_crm_goldford_config(
    extractions: dict,
    resource_names: list,
    Km_by_resource: dict,
    *,
    maintenance_by_species: dict | None = None,
    dilution: float = 0.0,
    feed_by_resource: dict | None = None,
    resource_loss_by_resource: dict | None = None,
) -> dict:
    """
    Convert extractor outputs into a crm_goldford config dict.

    extractions structure per species:
      {
        sp: {
          "yields":     {ex: Y_gDW_per_mmol},
          "byproducts": {ex_consumed: {ex_produced: secretion_flux}},  # fluxes
          "uptakes":    {ex: uptake_flux},                             # fluxes (Vmax)
        }, ...
      }
    """
    species_names = list(extractions.keys())
    r_idx = {r: j for j, r in enumerate(resource_names)}

    # 1) yields & Vmax (resource_uptakes)
    yields_map = {}
    vmax_map = {}
    for sp, blk in extractions.items():
        y = blk.get("yields", {})
        u = blk.get("uptakes", {})
        yields_map[sp] = {r: float(y.get(r, 0.0)) for r in resource_names}
        vmax_map[sp]   = {r: float(u.get(r, 0.0)) for r in resource_names}

    # 2) byproducts as produced-per-consumed (mmol/mmol), i.e., B[s,i,j]
    #    Convert raw secretion fluxes to ratios by dividing by the actual uptake of the consumed resource.
    B_map = {}
    for sp, blk in extractions.items():
        B_map[sp] = {}
        upt = blk.get("uptakes", {})
        for consumed, prod_fluxes in blk.get("byproducts", {}).items():
            v_cons = float(upt.get(consumed, 0.0))
            if v_cons <= 1e-12 or consumed not in r_idx:
                continue
            # normalize each produced flux by v_cons
            B_map[sp][consumed] = {
                produced: float(flux) / v_cons
                for produced, flux in prod_fluxes.items()
                if produced in r_idx and float(flux) > 0.0
            }

    # 3) Km (allow simple {resource: value})
    Km_cfg = {r: float(Km_by_resource.get(r, 0.5)) for r in resource_names}

    # 4) maintenance (defaults if not given)
    maint = {sp: float((maintenance_by_species or {}).get(sp, 0.02)) for sp in species_names}

    # 5) dilution/feed/loss
    feed = {r: float((feed_by_resource or {}).get(r, 0.0)) for r in resource_names}
    rloss = {r: float((resource_loss_by_resource or {}).get(r, 0.0)) for r in resource_names}

    # NOTE: crm_goldford config_schema allows either "resource_names" or "resources_names"
    cfg = {
        "species_names": species_names,
        "resource_names": resource_names,
        "yields": yields_map,
        "resource_uptakes": vmax_map,
        "maintenance": maint,
        "Km": Km_cfg,
        "byproducts": B_map,
        "dilution": float(dilution),
        "feed": feed,
        "resource_loss": rloss,
    }
    return cfg


def make_initial_state(species_names: list, resource_names: list,
                       n0_by_species: dict, R0_by_resource: dict) -> dict:
    # strategies are ignored by crm_goldford; include zeros for compatibility
    strategies = {s: {r: 0.0 for r in resource_names} for s in species_names}
    return {
        "species": {s: float(n0_by_species.get(s, 0.0)) for s in species_names},
        "concentrations": {r: float(R0_by_resource.get(r, 0.0)) for r in resource_names},
        "strategies": strategies,
    }

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


#=================================================
# Trying new way to get the yields, uptakes, and byproducts
#=================================================
def extract_yields_agora(
    model,
    resource_exchange_ids: List[str],
    *,
    biomass_rxn_id: str,
    use_pfba: bool = False,
) -> Dict[str, float]:
    """Yield per resource (gDW/mmol) with exchange opened fully (no uptake cap)."""
    yields: Dict[str, float] = {}
    for ex_id in resource_exchange_ids:
        with model as m:
            if ex_id not in m.reactions:
                yields[ex_id] = 0.0
                continue
            rxn = m.reactions.get_by_id(ex_id)
            rxn.lower_bound = -1000.0  # fully open uptake

            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol = model.optimize()

            if sol.status != "optimal":
                yields[ex_id] = 0.0
                continue

            v = float(sol.fluxes.get(ex_id, 0.0))
            uptake = max(0.0, -v)  # uptake is negative
            mu = float(sol.objective_value or 0.0)
            print(mu)
            yields[ex_id] = (mu / uptake) if uptake > 1e-12 else 0.0
    return yields


def extract_byproducts_agora(
    model,
    resource_exchange_ids: List[str],
    *,
    biomass_rxn_id: str,
    use_pfba: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Secreted exchange fluxes (mmol/gDW/h) when growing on each resource, no uptake cap."""
    out: Dict[str, Dict[str, float]] = {}
    for ex_id in resource_exchange_ids:
        with model as m:
            if ex_id not in m.reactions:
                out[ex_id] = {}
                continue
            rxn = m.reactions.get_by_id(ex_id)
            rxn.lower_bound = -1000.0

            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol = _solve(m, use_pfba)

            if sol.status != "optimal":
                out[ex_id] = {}
                continue

            secreted: Dict[str, float] = {}
            for ex in m.exchanges:
                f = float(sol.fluxes.get(ex.id, 0.0))
                if f > 1e-9:  # positive flux = secretion
                    secreted[ex.id] = f
            out[ex_id] = secreted
    return out


def extract_uptake_rates_agora(
    model,
    resource_exchange_ids: List[str],
    *,
    biomass_rxn_id: str,
    use_pfba: bool = False,
) -> Dict[str, float]:
    """Actual uptake rates (mmol/gDW/h) with exchange opened fully (no cap)."""
    uptakes: Dict[str, float] = {}
    for ex_id in resource_exchange_ids:
        with model as m:
            if ex_id not in m.reactions:
                uptakes[ex_id] = 0.0
                continue
            rxn = m.reactions.get_by_id(ex_id)
            rxn.lower_bound = -1000.0

            m.objective = m.reactions.get_by_id(biomass_rxn_id)
            sol = _solve(m, use_pfba)

            if sol.status != "optimal":
                uptakes[ex_id] = 0.0
                continue

            v = float(sol.fluxes.get(ex_id, 0.0))
            uptakes[ex_id] = max(0.0, -v)
    return uptakes