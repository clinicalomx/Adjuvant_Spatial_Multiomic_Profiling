import numpy as np
from scipy.integrate import simpson
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed
from itertools import combinations


class Spatial2DDensityCorrelation:

    def __init__(self, adata, phenotypeLabel, compartmentLabel,
                 bandwidth=100.0, grid_step=5.0, n_jobs=-1, verbose=True,
                 backend="threading"):
        self.adata = adata
        self.phenotypeLabel = phenotypeLabel
        self.compartmentLabel = compartmentLabel
        self.bandwidth = bandwidth
        self.grid_step = grid_step
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend

        self.KDEs = {}                 # (phenotype, compartment) -> fitted KDE or [] if too few cells
        self.densityGrids = {}         # (phenotype, compartment) -> Z, shape (ny, nx)
        self.compartmentGrids = {}     # compartment -> (x_axis, y_axis)
        self.results = {}              # (compartment, p1, p2) -> JSD value (float or NaN)

        self.Xmin = self.Xmax = self.Ymin = self.Ymax = None

        if (self.phenotypeLabel not in self.adata.obs.columns or
                self.compartmentLabel not in self.adata.obs.columns):
            raise KeyError(
                f'"{self.phenotypeLabel}" or "{self.compartmentLabel}" '
                f'not found in adata.obs'
            )

    def computeJSDAllCellTypes(self):
        for compartment in self.adata.obs[self.compartmentLabel].unique():
            if self.verbose:
                print(f"Computing for {compartment}")
            self._processCompartment(compartment)

    def returnXYMinMax(self):
        return np.array([self.Xmin, self.Xmax, self.Ymin, self.Ymax])

    def _processCompartment(self, compartment):
        obs = self.adata.obs
        comp_mask = (obs[self.compartmentLabel] == compartment)
        phenotypes = obs.loc[comp_mask, self.phenotypeLabel].unique()

        xy_by_p = {p: self._xy(compartment, p) for p in phenotypes}

        valid = [p for p, xy in xy_by_p.items() if len(xy) > 4]
        insufficient = [p for p in phenotypes if p not in valid]

        for p in insufficient:
            self.KDEs[(p, compartment)] = []
        for p1 in insufficient:
            for p2 in phenotypes:
                if p1 != p2:
                    self.results[(compartment, p1, p2)] = np.nan
                    self.results[(compartment, p2, p1)] = np.nan

        if len(valid) < 2:
            return

        all_xy = np.vstack([xy_by_p[p] for p in valid])
        xmin, ymin = all_xy.min(axis=0)
        xmax, ymax = all_xy.max(axis=0)

        x_axis = np.arange(xmin, xmax + self.grid_step, self.grid_step)
        y_axis = np.arange(ymin, ymax + self.grid_step, self.grid_step)
        Xg, Yg = np.meshgrid(x_axis, y_axis, indexing="xy")
        grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
        grid_shape = Xg.shape  # (ny, nx)

        self.compartmentGrids[compartment] = (x_axis, y_axis)

        bw = self.bandwidth
        fit_eval = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(fit_and_eval_kde)(xy_by_p[p], grid_pts, grid_shape, bw)
            for p in valid
        )

        Z_by_p = {}
        for p, (kde, Z) in zip(valid, fit_eval):
            self.KDEs[(p, compartment)] = kde
            self.densityGrids[(p, compartment)] = Z
            Z_by_p[p] = Z

        pairs = list(combinations(valid, 2))
        bboxes = [pair_bbox(xy_by_p[p1], xy_by_p[p2]) for p1, p2 in pairs]

        jsd_values = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(jsd_on_grid)(Z_by_p[p1], Z_by_p[p2], x_axis, y_axis, bbox)
            for (p1, p2), bbox in zip(pairs, bboxes)
        )

        for (p1, p2), val, bbox in zip(pairs, jsd_values, bboxes):
            self.results[(compartment, p1, p2)] = val
            self.results[(compartment, p2, p1)] = val
            self.Xmin, self.Xmax, self.Ymin, self.Ymax = bbox  # legacy "last pair"


    def _xy(self, compartment, phenotype):
        obs = self.adata.obs
        mask = (
            (obs[self.compartmentLabel] == compartment) &
            (obs[self.phenotypeLabel] == phenotype)
        )
        return obs.loc[mask, ["x", "y"]].to_numpy(dtype=float)


# Fit KDEs, parallelised
def fit_and_eval_kde(xy, grid_pts, grid_shape, bandwidth):
    kde = KernelDensity(
        kernel="gaussian", bandwidth=bandwidth,
        algorithm="kd_tree", leaf_size=100, atol=1e-9,
    ).fit(xy)
    Z = np.exp(kde.score_samples(grid_pts)).reshape(grid_shape)
    return kde, Z


#  Bounding box of the union of two XY point sets.
def pair_bbox(xy1, xy2):
    xmin = float(min(xy1[:, 0].min(), xy2[:, 0].min()))
    xmax = float(max(xy1[:, 0].max(), xy2[:, 0].max()))
    ymin = float(min(xy1[:, 1].min(), xy2[:, 1].min()))
    ymax = float(max(xy1[:, 1].max(), xy2[:, 1].max()))
    return (xmin, xmax, ymin, ymax)


def jsd_on_grid(Z1_full, Z2_full, x_axis, y_axis, bbox):
    """
    Jensen-Shannon distance (sqrt of JSD divergence, base-2 logs) between two
    density grids, integrated over bbox.
    """
    xmin, xmax, ymin, ymax = bbox
    ix = np.where((x_axis >= xmin) & (x_axis <= xmax))[0]
    iy = np.where((y_axis >= ymin) & (y_axis <= ymax))[0]

    if len(ix) < 2 or len(iy) < 2:
        return np.nan

    Z1 = Z1_full[np.ix_(iy, ix)]
    Z2 = Z2_full[np.ix_(iy, ix)]
    x_sub = x_axis[ix]
    y_sub = y_axis[iy]

    # JSD = 0.5 KL(P||M) + 0.5 KL(Q||M),   M = 0.5(P+Q)
    M = 0.5 * (Z1 + Z2)
    eps = 1e-30
    with np.errstate(divide="ignore", invalid="ignore"):
        kl1 = Z1 * np.log2((Z1 + eps) / (M + eps))
        kl1 = np.where(Z1 > 0.0, kl1, 0.0)
        kl2 = Z2 * np.log2((Z2 + eps) / (M + eps))
        kl2 = np.where(Z2 > 0.0, kl2, 0.0)

    int_kl1 = simpson(simpson(kl1, x=x_sub, axis=1), x=y_sub)
    int_kl2 = simpson(simpson(kl2, x=x_sub, axis=1), x=y_sub)

    js_div = 0.5 * (int_kl1 + int_kl2)
    return float(np.sqrt(max(js_div, 0.0)))

