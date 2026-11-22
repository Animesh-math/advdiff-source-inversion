"""Application2 tracer-transport adapter around the shared solvers.

Nothing in ``Src/`` or ``Applications/Application1`` is modified.  This module

* builds the transport ``mesh_data`` for an Application2 mesh level and its own
  Navier-Stokes velocity (association enforced by ``mesh_data_config``);
* exposes the clean-air Dirichlet set under the boundary name ``"left"``,
  because the shared objective/adjoint/initializer code
  (``Src/Objective/gradient_test.py``, ``Application1/*``) hard-codes
  ``dirichlet_tags=["left"]``.  For Application2 ``"left"`` is therefore an
  ALIAS of the three supply vents and has nothing to do with the wall x = 0
  (which is ``wall_left``);
* assembles the semi-discrete operator with the shared forward solver's own
  assembly routine and time-steps it with the same RK4 formula using a SPARSE
  matrix-vector product (``SparseTransport``), which is what makes the fine mesh
  affordable.  ``validate_readiness.py`` checks it against
  ``gradient_test.run_forward`` on the coarse mesh;
* fixes the sensors (coarse-mesh nodes) and builds the P1 interpolation operator
  that samples any mesh level at those coordinates.

Module-name collision: Application1 and Application2 both have a
``mesh_data_config`` and a ``navier_stokes_config``.  Application1's code imports
them by bare name, so here the Application2 versions are loaded under the
private names ``application2_mesh_data_config`` /
``application2_navier_stokes_config`` and Application1's directory is put first
on ``sys.path`` before any Application1 module is imported.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

APP2_DIR = Path(__file__).resolve().parent
REPO = APP2_DIR.parents[1]
APP1_DIR = REPO / "Applications" / "Application1"
SRC_SOLVERS = REPO / "Src" / "Solvers"

if str(APP2_DIR) not in sys.path:
    sys.path.append(str(APP2_DIR))

import room_geometry_config as geo  # noqa: E402  (unique to Application2)
import transport_config as tc  # noqa: E402


def _load_private(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mdc = _load_private("application2_mesh_data_config", APP2_DIR / "mesh_data_config.py")
nsc = _load_private("application2_navier_stokes_config", APP2_DIR / "navier_stokes_config.py")

OUTPUT_ROOT = APP2_DIR / "output"
FLOW_DIR = OUTPUT_ROOT / tc.FLOW_VERSION
DATASET_DIR = OUTPUT_ROOT / tc.DATASET_VERSION

DIRICHLET_ALIAS = "left"


# --------------------------------------------------------------------------- #
# Application1 / shared-solver access
# --------------------------------------------------------------------------- #
class _App1:
    """Lazily imported Application1 modules (unmodified)."""

    _loaded = None

    @classmethod
    def load(cls):
        if cls._loaded is not None:
            return cls._loaded
        if "mesh_data_config" in sys.modules and Path(
                sys.modules["mesh_data_config"].__file__).resolve().parent != APP1_DIR:
            raise ImportError(
                "a non-Application1 'mesh_data_config' is already imported; import transport_model before any "
                "bare 'import mesh_data_config' (Application2's is available as transport_model.mdc)")
        sys.path.insert(0, str(APP1_DIR))
        import mesh_data_config as app1_mdc  # noqa: F401

        if Path(app1_mdc.__file__).resolve().parent != APP1_DIR:
            raise ImportError(f"mesh_data_config resolved to {app1_mdc.__file__}, expected Application1's")
        import run_optimization as ro
        import post_init_optimizers as pio
        import sensitivity_initialization as sinit
        from mesh_admissibility import MeshAdmissibility

        cls._loaded = type("App1Modules", (), {
            "ro": ro, "gt": ro.gt, "pio": pio, "sinit": sinit, "obs_op": ro.obs_op,
            "MeshAdmissibility": MeshAdmissibility, "Optimizer": ro.Optimizer,
        })
        return cls._loaded


def app1():
    return _App1.load()


# --------------------------------------------------------------------------- #
# mesh_data / PDE dictionary
# --------------------------------------------------------------------------- #
def build_transport_mesh_data(level: str, flow_dir: Path = FLOW_DIR) -> dict:
    """Application2 mesh_data plus the clean-air Dirichlet alias and the velocity field."""
    md = mdc.build_mesh_data(level, flow_dir)
    velocity = mdc.load_velocity_nodes(level, md, flow_dir)      # raises on any mesh/field mismatch
    md["coord"] = np.ascontiguousarray(md["coord"][:, :2])
    names = list(tc.CLEAN_AIR_DIRICHLET_NAMES)
    nodes = np.unique(np.concatenate([md["boundary_nodes_by_name"][n] for n in names]))
    edges = np.concatenate([md["boundary_edges_by_name"][n] for n in names], axis=0)
    if DIRICHLET_ALIAS in md["boundary_nodes_by_name"]:
        raise ValueError(f"mesh already defines a boundary named {DIRICHLET_ALIAS!r}")
    md["boundary_nodes_by_name"][DIRICHLET_ALIAS] = nodes
    md["boundary_edges_by_name"][DIRICHLET_ALIAS] = edges
    md["clean_air_dirichlet_names"] = names
    md["clean_air_dirichlet_nodes"] = nodes
    md["dirichlet_alias_note"] = (f"boundary name {DIRICHLET_ALIAS!r} is an alias of {names} (the shared solvers "
                                  f"hard-code dirichlet_tags=['left']); it is not the wall x = 0")
    md["velocity_nodes"] = velocity
    md["velocity_file"] = str(mdc.velocity_npz_file(level, flow_dir))
    md["velocity_sha256"] = mdc.sha256_file(mdc.velocity_npz_file(level, flow_dir))
    return md


def build_pde(level: str, mesh_data: dict, *, k: float | None = None, dt: float | None = None,
              tfinal: float | None = None, sigma2: float | None = None) -> dict:
    """PDE dictionary in the layout the shared forward/adjoint code expects."""
    return {
        "lx": geo.DOMAIN_X_MAX,                    # ignored by the solver (geometry is the mesh)
        "ly": geo.ROOM_Y_MAX,
        "k": float(tc.DIFFUSIVITY_K if k is None else k),
        "vx": 0.0,                                  # constant-velocity fallback, unused: velocity_nodes is set
        "vy": 0.0,
        "dt": float(tc.DT[level] if dt is None else dt),
        "tfinal": float(tc.TFINAL if tfinal is None else tfinal),
        "jprint": 1,
        "ngauss": tc.NGAUSS,
        "sigma2": float(tc.SIGMA2 if sigma2 is None else sigma2),
        "objective_time_rule": tc.OBJECTIVE_TIME_RULE,
        "velocity_nodes": mesh_data["velocity_nodes"],
        "velocity_field": {"path": mesh_data["velocity_file"], "sha256": mesh_data["velocity_sha256"],
                           "mesh_level": level},
    }


def num_steps(dt: float, tfinal: float) -> int:
    """Number of RK4 steps the shared forward solver takes (its own formula)."""
    return int(np.floor((tfinal + 0.5 * dt) / dt))


def gaussian_source(z, sigma2: float):
    """The Application1 source function S(x, y) for a flat control (unchanged formula)."""
    from_control = app1().ro.source_from_control
    return from_control(np.asarray(z, dtype=np.float64), r2=float(sigma2))


# --------------------------------------------------------------------------- #
# Semi-discrete operator (shared assembly) + sparse RK4
# --------------------------------------------------------------------------- #
class SparseTransport:
    """M_lumped du/dt = b - K u with the shared solver's operator, stepped sparsely.

    ``K``, the lumped mass and the Dirichlet treatment come from
    ``Src/Solvers/adv_diff_forward.py`` (``_assemble_system``,
    ``_apply_homogeneous_dirichlet``) exactly as ``adv_diff_dyn`` builds them;
    only the matrix-vector product is sparse.  The RK4 update and the Dirichlet
    reset after every step are the same formulas.
    """

    def __init__(self, mesh_data: dict, pde: dict):
        gt = app1().gt
        fwd = gt._FWD
        self.coord = np.asarray(mesh_data["coord"], dtype=np.float64)
        self.connections = np.asarray(mesh_data["connections"], dtype=int)
        self.npoin, self.nelem = self.coord.shape[0], self.connections.shape[0]
        self.pde = pde
        self._gt = gt
        params = fwd.PDEParams(k=pde["k"], velocity=(pde["vx"], pde["vy"]), dt=pde["dt"], tfinal=pde["tfinal"],
                               ngauss=pde["ngauss"], dirichlet_tags=[DIRICHLET_ALIAS],
                               velocity_nodes=pde.get("velocity_nodes"))
        kmatrix, _zero_source, mass = fwd._assemble_system(
            npoin=self.npoin, nelem=self.nelem, coord=self.coord, connections=self.connections,
            source_func=lambda x, y: 0.0, params=params)
        self.dirichlet_nodes = fwd._dirichlet_nodes_from_mesh_data(mesh_data, [DIRICHLET_ALIAS])
        fwd._apply_homogeneous_dirichlet(kmatrix, _zero_source, self.dirichlet_nodes)
        mass[self.dirichlet_nodes] = 1.0
        if np.any(mass <= 0.0):
            raise ValueError("non-positive lumped mass")
        self.K = sp.csr_matrix(kmatrix)
        del kmatrix
        self.mass = mass
        self.inv_mass = 1.0 / mass

    def load_vector(self, source_func) -> np.ndarray:
        b = self._gt.assemble_load_vector(self.npoin, self.nelem, self.coord, self.connections,
                                          self.pde["ngauss"], source_func)
        b[self.dirichlet_nodes] = 0.0
        return b

    def spectral_radius(self) -> float:
        """Largest |eigenvalue| of M_lumped^{-1} K (the operator RK4 actually integrates)."""
        A = sp.diags(self.inv_mass) @ self.K
        lam = spla.eigs(A, k=1, which="LM", return_eigenvectors=False, maxiter=50000, tol=1.0e-8)
        return float(np.abs(lam[0]))

    def solve(self, load: np.ndarray, *, sample_every: int = 1, sampler=None, snapshot_steps=(),
              progress=None):
        """Integrate from u = 0.  Returns ``(samples, snapshots)``.

        ``samples[n]`` is ``sampler @ u`` (or ``u`` itself if ``sampler`` is None)
        after ``n * sample_every`` steps, n = 0 .. nsteps // sample_every.
        ``snapshots`` maps each requested step index to the full state.
        """
        dt = float(self.pde["dt"])
        nsteps = num_steps(dt, float(self.pde["tfinal"]))
        K, inv_mass, dn = self.K, self.inv_mass, self.dirichlet_nodes
        u = np.zeros(self.npoin)
        take = (lambda v: sampler @ v) if sampler is not None else (lambda v: v.copy())
        first = take(u)
        samples = np.zeros((nsteps // sample_every + 1,) + first.shape)
        samples[0] = first
        wanted, snapshots = set(int(s) for s in snapshot_steps), {}
        if 0 in wanted:
            snapshots[0] = u.copy()

        def rhs(v):
            val = inv_mass * (load - K @ v)
            return val

        for step in range(1, nsteps + 1):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            u[dn] = 0.0
            if step % sample_every == 0:
                samples[step // sample_every] = take(u)
            if step in wanted:
                snapshots[step] = u.copy()
            if progress is not None and step % progress[0] == 0:
                progress[1](step, nsteps)
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("non-finite state: the explicit time step is unstable")
        return samples, snapshots


def cell_peclet(mesh_data: dict, k: float) -> dict:
    """Cell Peclet numbers |v_centroid| * h_max_edge / (2 k), area-weighted statistics."""
    coord, tri, vel = mesh_data["coord"], mesh_data["connections"], mesh_data["velocity_nodes"]
    edges = np.stack([np.linalg.norm(coord[tri[:, i]] - coord[tri[:, (i + 1) % 3]], axis=1) for i in range(3)], 1)
    x, y = coord[tri, 0], coord[tri, 1]
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    pe = np.linalg.norm(vel[tri].mean(axis=1), axis=1) * edges.max(axis=1) / (2.0 * float(k))
    w = area / area.sum()
    order = np.argsort(pe)
    cw = np.cumsum(w[order])

    def quantile(p):
        return float(pe[order][min(np.searchsorted(cw, p), pe.size - 1)])

    return {"k": float(k), "max": float(pe.max()), "area_median": quantile(0.5), "area_q90": quantile(0.9),
            "area_q99": quantile(0.99), "area_fraction_above_1": float(w[pe > 1.0].sum()),
            "area_fraction_above_2": float(w[pe > 2.0].sum()),
            "edge_min_m": float(edges.min()), "edge_median_m": float(np.median(edges)),
            "edge_max_m": float(edges.max()), "speed_max_m_per_s": float(np.linalg.norm(vel, axis=1).max())}


# --------------------------------------------------------------------------- #
# Sensors
# --------------------------------------------------------------------------- #
def boundary_node_ids(mesh_data: dict) -> np.ndarray:
    return np.unique(np.asarray(mesh_data["boundary_lines"], dtype=int).reshape(-1))


def fix_sensors(coarse_mesh_data: dict, layout: str | None = None) -> dict:
    """Snap the geometric sensor targets of a layout to coarse-mesh nodes.

    Uses only the room geometry and the coarse mesh: no concentration field, no
    source information.  Boundary nodes (walls, beds, partitions, openings --
    hence also every clean-air Dirichlet node) are never selected, so each
    sampling point lies inside the fluid.  ``node_ids`` index the coarse mesh;
    ``coordinates`` are the sampling points used on every mesh level.
    """
    layout = tc.ACTIVE_SENSOR_LAYOUT if layout is None else layout
    coord = coarse_mesh_data["coord"]
    allowed = np.setdiff1d(np.arange(coord.shape[0]), boundary_node_ids(coarse_mesh_data))
    segments = geo.all_boundary_segments()
    rows, used = [], set()
    for target in tc.sensor_targets(layout):
        x, y = target["intended_xy"]
        d = np.hypot(coord[allowed, 0] - x, coord[allowed, 1] - y)
        node = int(allowed[int(np.argmin(d))])
        dist = float(d.min())
        if dist > tc.SENSOR_SNAP_TOLERANCE:
            raise ValueError(f"sensor {target['label']}: nearest interior coarse node is {dist:.3f} m away")
        if node in used:
            raise ValueError(f"sensor {target['label']}: snapped to an already used node {node}")
        used.add(node)
        cx, cy = float(coord[node, 0]), float(coord[node, 1])
        row = dict(target)
        row.update({"coarse_node_id": node, "xy": [cx, cy], "snap_distance_m": dist,
                    "distance_to_boundary_m": float(geo.distance_to_boundary(cx, cy)[0]), "on_boundary": False})
        if target["wall"] is not None:
            offs = []
            for p, q, name in segments:
                if name != target["wall"]:
                    continue
                p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
                t = np.clip(np.dot([cx - p[0], cy - p[1]], q - p) / np.dot(q - p, q - p), 0.0, 1.0)
                offs.append(float(np.hypot(*(np.array([cx, cy]) - (p + t * (q - p))))))
            row["wall_offset_m"] = min(offs)
            row["intended_wall_offset_m"] = tc.SENSOR_WALL_OFFSET
            wp = target["wall_point"]
            row["along_wall_shift_m"] = float(abs((cx - wp[0]) * target["inward_normal"][1]
                                                 - (cy - wp[1]) * target["inward_normal"][0]))
        rows.append(row)
    return {
        "layout": layout,
        "active_layout": layout == tc.ACTIVE_SENSOR_LAYOUT,
        "selection_rule": tc.SENSOR_LAYOUT_RULES[layout],
        "num_sensors": len(rows),
        "snap_tolerance_m": tc.SENSOR_SNAP_TOLERANCE,
        "coarse_mesh_sha256": coarse_mesh_data["mesh_sha256"],
        "sensors": rows,
        "labels": [r["label"] for r in rows],
        "node_ids": [r["coarse_node_id"] for r in rows],
        "coordinates": [r["xy"] for r in rows],
    }


def interpolation_matrix(mesh_data: dict, points) -> tuple:
    """P1 finite-element interpolation operator at physical points.

    Returns ``(P, info)`` with ``P`` a sparse (n_points, n_nodes) matrix of
    barycentric weights of the containing triangle.  Raises if a point is not
    inside the mesh.
    """
    import matplotlib.tri as mtri

    coord, tri = mesh_data["coord"], mesh_data["connections"]
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    finder = mtri.Triangulation(coord[:, 0], coord[:, 1], tri).get_trifinder()
    cells = np.asarray(finder(pts[:, 0], pts[:, 1]), dtype=int)
    rows, cols, vals, info = [], [], [], []
    for i, (p, cell) in enumerate(zip(pts, cells)):
        if cell < 0:                                   # exactly on an edge/vertex: brute-force with tolerance
            cell = _brute_force_cell(coord, tri, p)
        nodes = tri[cell]
        a, b, c = coord[nodes]
        T = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
        l1, l2 = np.linalg.solve(T, p - a)
        w = np.array([1.0 - l1 - l2, l1, l2])
        if w.min() < -1.0e-9:
            raise ValueError(f"point {p} is outside its located triangle (weights {w})")
        w = np.clip(w, 0.0, None)
        w /= w.sum()
        rows += [i, i, i]
        cols += [int(n) for n in nodes]
        vals += [float(v) for v in w]
        info.append({"cell": int(cell), "nodes": [int(n) for n in nodes], "weights": [float(v) for v in w]})
    P = sp.csr_matrix((vals, (rows, cols)), shape=(pts.shape[0], coord.shape[0]))
    return P, info


def _brute_force_cell(coord, tri, p, tol: float = 1.0e-9) -> int:
    a, b, c = coord[tri[:, 0]], coord[tri[:, 1]], coord[tri[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    l1 = ((p[0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (p[1] - a[:, 1])) / det
    l2 = ((b[:, 0] - a[:, 0]) * (p[1] - a[:, 1]) - (p[0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
    ok = np.flatnonzero((l1 >= -tol) & (l2 >= -tol) & (l1 + l2 <= 1.0 + tol))
    if ok.size == 0:
        raise ValueError(f"point {p} is not inside the mesh")
    return int(ok[0])


# --------------------------------------------------------------------------- #
# Small utilities shared by the pipeline scripts
# --------------------------------------------------------------------------- #
def array_sha256(a) -> str:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")
