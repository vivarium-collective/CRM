from process_bigraph import Process
from process_bigraph.emitter import emitter_from_wires
import numpy as np
from scipy.integrate import solve_ivp

class CommunityModelsProcess(Process):
    """
    Vivarium Process wrapper for the CommunityModels suite:
      - 'mcrm'     : Modern CRM variant with byproducts (your _mcrm_dynamics)
      - 'gibbs'    : Gibbs-style minimum rule model (your _gibbs_dynamics)
      - 'micrm'    : MiCRM (Goldford/Wu-style with leakage & crossfeeding)
      - 'adaptive' : Picciani–Mori adaptive CRM (N, C, A dynamics)
      - 'classical': MacArthur CRM (logistic/external/tilman resource modes)

    This process expects state maps (species, concentrations, allocations, etc.)
    and integrates ODEs for a provided interval, returning deltas.
    """

    # -------- Config schema (same style as your prior Process) --------
    config_schema = {
        "model": "string",       # 'mcrm' | 'gibbs' | 'micrm' | 'adaptive' | 'classical'
        "params": "map[Any]",    # parameter dict(s) as required by each model (see below)
        "rtol": "float",         # ODE solver relative tolerance
        "atol": "float",         # ODE solver absolute tolerance
        "method": "string",      # ODE method, e.g. "RK45" or "BDF"
        "nonneg_clip": "boolean" # clip negatives after integration
    }

    # ------------------------ Lifecycle ------------------------
    def initialize(self, config):
        # Required
        self.model = config["model"].lower()
        self.params = dict(config["params"])

        # Solver knobs
        self.rtol = float(config.get("rtol", 1e-6))
        self.atol = float(config.get("atol", 1e-9))
        self.method = str(config.get("method", "RK45"))
        self.nonneg_clip = bool(config.get("nonneg_clip", True))

        # Build an internal CommunityModels engine with only the selected params
        self.cm = CommunityModels(
            mcrm_params=self.params if self.model == "mcrm" else None,
            gibbs_params=self.params if self.model == "gibbs" else None,
            micrm_params=self.params if self.model == "micrm" else None,
            adaptive_params=self.params if self.model == "adaptive" else None,
            classical_params=self.params if self.model == "classical" else None,
        )

        # Figure out which state variables we expect from stores, and how to pack/unpack
        self._compile_variable_layout()

    def _compile_variable_layout(self):
        """
        Establish the variable naming and vector layout for each model so we can:
          - read current state from stores -> vector y0
          - integrate -> y1
          - map back to store fields and return deltas
        """
        m = self.model
        self.var_order = []   # list of (field_name, keys_list, slice(start, end))
        self.size = 0

        if m == "classical":
            # Needs 'species' (N) and 'concentrations' (R)
            # The params contain c (S,R), which defines counts if initial_N/R not provided.
            c = np.asarray(self.params["c"], float)
            S, R = c.shape

            # We'll read names from state if available; otherwise autogenerate.
            self.species_names = self.params.get("species_names", [f"S{i+1}" for i in range(S)])
            self.resource_names = self.params.get("resource_names", [f"R{j+1}" for j in range(R)])

            self.var_order.append(("species", self.species_names, slice(self.size, self.size + S))); self.size += S
            self.var_order.append(("concentrations", self.resource_names, slice(self.size, self.size + R))); self.size += R

            # Bind dynamics closures (vectorized Euler is internal; we’ll integrate ODE form using your algebra)
            tau = float(self.params["tau"])
            mvec = np.asarray(self.params["m"], float)
            w    = np.asarray(self.params["w"], float)
            r    = np.asarray(self.params["r"], float)
            K    = np.asarray(self.params.get("K", self.params.get("kappa")), float)
            mode = self.params.get("resource_mode", "logistic")

            def classical_rhs(t, y):
                N = y[:S]; R = y[S:S+R]
                growth_input = (c * w.reshape(1, -1)) @ R
                dN = (N / tau) * (growth_input - mvec)
                if mode in ("logistic", "external"):
                    consumption = (N @ c) * R
                else:
                    consumption = (N @ c)
                if mode == "logistic":
                    regeneration = (r / K) * (K - R) * R
                else:
                    regeneration = r * (K - R)
                dR = regeneration - consumption
                return np.concatenate([dN, dR])

            self._rhs = classical_rhs

        elif m == "micrm":
            # MiCRM expects N (species) and R (resources)
            # params keys: C (S,R), D (R,R), leakage (scalar or (R,)), rho (R,), tau (R,),
            #              w (S,), m (S,), g (scalar), R0 (R,), N0 (S,)
            C = np.asarray(self.params["C"], float)
            S, R = C.shape

            self.species_names  = self.params.get("species_names", [f"S{i+1}" for i in range(S)])
            self.resource_names = self.params.get("resource_names", [f"R{j+1}" for j in range(R)])

            # Vector order: first R (MiCRM code uses R then N), then N
            self.var_order.append(("concentrations", self.resource_names, slice(self.size, self.size + R))); self.size += R
            self.var_order.append(("species", self.species_names, slice(self.size, self.size + S))); self.size += S

            def micrm_rhs(t, x):
                return self.cm._micrm_dynamics(t, x)

            self._rhs = micrm_rhs

        elif m == "mcrm":
            # Your MCRM packs an arbitrary x with varIdx mapping.
            v = self.params["varIdx"]
            # We will expose species/resources if varIdx provides them.
            # Determine lengths from x0
            x0 = np.asarray(self.params["x0"], float)
            # Build ordered exposure if present:
            self.species_names  = self.params.get("species_names") or [f"S{i+1}" for i in range(len(v.get("species", [])))]
            self.resource_names = self.params.get("resource_names") or [f"R{j+1}" for j in range(len(v.get("resources", [])))]

            # We maintain the *native* layout for integration (x = whole vector)
            self.size = len(x0)
            self._mcrm_full = True

            def mcrm_rhs(t, x):
                return self.cm._mcrm_dynamics(t, x)

            self._rhs = mcrm_rhs

        elif m == "gibbs":
            # Gibbs uses vector order: first R (S resources), then N (K species).
            S = int(self.params["num_resources"])
            K = int(len(self.params["N0"]))

            self.resource_names = self.params.get("resource_names", [f"R{j+1}" for j in range(S)])
            self.species_names  = self.params.get("species_names",  [f"S{i+1}" for i in range(K)])

            self.var_order.append(("concentrations", self.resource_names, slice(self.size, self.size + S))); self.size += S
            self.var_order.append(("species", self.species_names, slice(self.size, self.size + K))); self.size += K

            def gibbs_rhs(t, x):
                return self.cm._gibbs_dynamics(t, x)

            self._rhs = gibbs_rhs

        elif m == "adaptive":
            # Adaptive Picciani–Mori: state = [N(S), C(R), A(S*R)]
            S, R = int(self.params["S"]), int(self.params["R"])
            self.species_names  = self.params.get("species_names",  [f"S{i+1}" for i in range(S)])
            self.resource_names = self.params.get("resource_names", [f"R{j+1}" for j in range(R)])

            self.var_order.append(("species", self.species_names, slice(self.size, self.size + S))); self.size += S
            self.var_order.append(("concentrations", self.resource_names, slice(self.size, self.size + R))); self.size += R
            self.var_order.append(("allocations", [f"A_{i+1}_{j+1}" for i in range(S) for j in range(R)],
                                   slice(self.size, self.size + S * R))); self.size += S * R

            def adaptive_rhs(t, x):
                return self.cm.adaptive_crm_dynamics(t, x)

            self._rhs = adaptive_rhs

        else:
            raise ValueError(f"Unknown model '{self.model}'")

    # ------------------------ I/O Spec ------------------------
    def inputs(self):
        # We only declare the keys that this model will read.
        keys = {}
        for name, names_list, _sl in self.var_order if hasattr(self, "var_order") else []:
            keys[name] = "map[float]"
        # Some models may rely solely on x0 inside params (MCRM). In that case,
        # var_order may be empty; we still allow empty inputs.
        return keys

    def outputs(self):
        # We return deltas for all keys we read
        out = {}
        for name, _, _ in self.var_order if hasattr(self, "var_order") else []:
            out[name + "_delta"] = "map[float]"
        # MCRM special: if we didn't expose species/resources via var_order, return empty outputs
        return out

    # ------------------------ Utilities ------------------------
    def _read_state_vector(self, state):
        """
        Build y0 in the order required by self._rhs, based on current store values.
        For MCRM, if using full x0 layout, we start from params['x0'] and overwrite
        species/resources slices (if provided in varIdx) from the state.
        """
        if self.model == "mcrm" and getattr(self, "_mcrm_full", False):
            y0 = np.array(self.params["x0"], dtype=float)
            v = self.params["varIdx"]
            # If user provided species/resources in state, overwrite those slices:
            if "species" in state and "species" in v:
                y0[v["species"]] = np.array([state["species"].get(n, y0[v["species"][i]])
                                             for i, n in enumerate(self.species_names)], float)
            if "concentrations" in state and "resources" in v:
                y0[v["resources"]] = np.array([state["concentrations"].get(n, y0[v["resources"][j]])
                                               for j, n in enumerate(self.resource_names)], float)
            return y0

        # General case: concatenate maps as defined by var_order
        parts = []
        for field, names, _sl in self.var_order:
            vals = [float(state[field][nm]) for nm in names]
            parts.append(np.asarray(vals, float))
        return np.concatenate(parts) if parts else np.array([], float)

    def _split_vector_to_maps(self, y0, y1):
        """
        Convert y0,y1 vectors back into per-field maps of deltas.
        """
        deltas = {}
        for field, names, slc in self.var_order:
            before = y0[slc]; after = y1[slc]
            delta_map = {nm: float(max(0.0, a) - b) if self.nonneg_clip else float(a - b)
                         for nm, a, b in zip(names, after, before)}
            deltas[field + "_delta"] = delta_map
        return deltas

    # ------------------------ Update ------------------------
    def update(self, state, interval):
        """
        Integrate ODEs for 'interval' time units and return deltas for exposed fields.
        """
        # 1) Build initial vector
        y0 = self._read_state_vector(state)

        # 2) Integrate
        if y0.size == 0:
            # Nothing to do—e.g., MCRM with no exposed fields
            return {}

        sol = solve_ivp(self._rhs, [0.0, float(interval)], y0,
                        method=self.method, t_eval=[float(interval)],
                        rtol=self.rtol, atol=self.atol)

        y1 = sol.y[:, -1] if sol.y.ndim == 2 else sol.y

        # 3) Clip non-negatives if requested
        if self.nonneg_clip:
            y1 = np.maximum(y1, 0.0)

        # 4) Map back to deltas
        return self._split_vector_to_maps(y0, y1)


# ------------------------ Emitter Helper ------------------------
def get_cm_emitter(state_keys):
    """
    Returns a standard emitter step spec for CommunityModels simulations.
    Only includes relevant state keys if present, mirroring your prior helper.
    """
    POSSIBLE_KEYS = {"species", "concentrations", "allocations", "global_time"}
    included = [k for k in POSSIBLE_KEYS if k in state_keys]
    return emitter_from_wires({k: [k] for k in included})