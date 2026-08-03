import numpy as np
from numbers import Number
from typing import Union
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

from anndata import AnnData
from typing import Union
import numpy as np
from pandas import DataFrame

def calculate_gcross(
    adata: AnnData,
    mark_key: str = None,
    library_key: str = None,
    spatial_key: str = "spatial",
    calculation_radii = None,
    n_interpolated_points=100,
    marks=None,
    edge_correction: Union[str, None] = None,
    copy: bool = False,
    gcross_key: str = "gcross",
    compute_labels: bool = True
) -> Union[DataFrame, None]:
    
    sample_gcross_results = []

    samples = sorted(adata.obs[library_key].unique())
    if marks is None:
        marks = sorted(adata.obs[mark_key].unique())

    for i, sample in enumerate(samples):
        sample_adata = adata[adata.obs[library_key] == sample]
        point_positions = sample_adata.obsm[spatial_key].copy()
        point_marks = sample_adata.obs[mark_key].to_numpy()

        gcross_results = gcross(
            point_locations=point_positions,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            edge_correction=edge_correction,
            something_edge_matrix=None,
            n_interpolated_points=n_interpolated_points,
            marks=marks
        )

        sample_gcross_results.append(gcross_results.reshape((len(marks)**2, len(calculation_radii))))

    sample_gcross_results = np.array(sample_gcross_results)
    
    if compute_labels:
        labels = [f"{m1}_@_to_@_{m2}" for m1 in marks for m2 in marks]
        if copy:
            return sample_gcross_results, labels
        else:
            adata.uns[f"{mark_key}_{gcross_key}"] = sample_gcross_results
            adata.uns[f"{mark_key}_{gcross_key}_labels"] = labels
            return None
    else:
        if copy:
            return sample_gcross_results
        else:
            adata.uns[f"{mark_key}_{gcross_key}"] = sample_gcross_results
    
    return sample_gcross_results

def calculate_gcross_inhomogeneous(
    adata: AnnData,
    mark_key: str = None,
    library_key: str = None,
    spatial_key: str = "spatial",
    intensity_key: str = "intensity",
    intensity_min: str = None,
    intensity_min_key: str = "intensity_min",
    calculation_radii = None,
    n_interpolated_points=100,
    marks=None,
    copy: bool = False,
    gcross_inhomogeneous_key: str = "gcross_inhomogeneous",
    compute_labels: bool = True
) -> Union[DataFrame, None]:
    
    sample_gcross_results = []

    samples = sorted(adata.obs[library_key].unique())
    if marks is None:
        marks = sorted(adata.obs[mark_key].unique())
    else:
        marks = sorted(marks)

    for i, sample in enumerate(samples):
        sample_adata = adata[adata.obs[library_key] == sample]
        point_positions = sample_adata.obsm[spatial_key].copy()
        point_marks = sample_adata.obs[mark_key].to_numpy()
        point_intensities = sample_adata.obs[intensity_key].to_numpy()
        if intensity_min is None:
            minimum_intensity = sample_adata.obs[intensity_min_key].to_numpy()

        gcross_results = gcross_inhomogeneous(
            point_locations=point_positions,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            point_intensities=point_intensities,
            minimum_intensity=minimum_intensity,
            n_interpolated_points=n_interpolated_points,
            marks=marks
        )

        sample_gcross_results.append(gcross_results.reshape((len(marks)**2, len(calculation_radii))))

    sample_gcross_results = np.array(sample_gcross_results)
    
    if compute_labels:
        labels = [f"{m1}_@_to_@_{m2}" for m1 in marks for m2 in marks]
        if copy:
            return sample_gcross_results, labels
        else:
            adata.uns[f"{mark_key}_{gcross_inhomogeneous_key}"] = sample_gcross_results
            adata.uns[f"{mark_key}_{gcross_inhomogeneous_key}_labels"] = labels
            return None
    else:
        if copy:
            return sample_gcross_results
        else:
            adata.uns[f"{mark_key}_{gcross_inhomogeneous_key}"] = sample_gcross_results
    
    return sample_gcross_results


def calculate_gcrossauc(
    adata: AnnData,
    mark_key: str = None,
    library_key: str = None,
    spatial_key: str = "spatial",
    calculation_radii = None,
    n_interpolated_points=100,
    marks=None,
    edge_correction: Union[str, None] = None,
    copy: bool = False,
    gcross_auc_key: str = "gcross_auc",
    compute_labels: bool = True
) -> Union[DataFrame, None]:
    
    sample_gcross_results, labels = calculate_gcross(
        adata=adata,
        mark_key=mark_key,
        library_key=library_key,
        spatial_key=spatial_key,
        calculation_radii=calculation_radii,
        n_interpolated_points=n_interpolated_points,
        marks=marks,
        edge_correction=edge_correction,
        copy=True,
        compute_labels=True
    )

    sample_gcrossauc_results = np.trapezoid(
        y=sample_gcross_results,
        x=calculation_radii,
        axis=2
    )

    if compute_labels:
        if copy:
            return sample_gcrossauc_results, labels
        else:
            adata.uns[f"{mark_key}_{gcross_auc_key}"] = sample_gcrossauc_results
            adata.uns[f"{mark_key}_{gcross_auc_key}_labels"] = labels
            return None
    else:
        if copy:
            return sample_gcrossauc_results
        else:
            adata.uns[f"{mark_key}_{gcross_auc_key}"] = sample_gcrossauc_results

def calculate_gcrossauc_inhomogeneous(
    adata: AnnData,
    mark_key: str = None,
    library_key: str = None,
    spatial_key: str = "spatial",
    intensity_key: str = "intensity",
    intensity_min: float = None,
    intensity_min_key: str = "intensity_min",
    calculation_radii = None,
    n_interpolated_points=100,
    marks=None,
    copy: bool = False,
    gcrossauc_inhomogeneous_key: str = "gcrossauc_inhomogeneous",
    compute_labels: bool = True
) -> Union[DataFrame, None]:
    
    sample_gcross_results, labels = calculate_gcross_inhomogeneous(
        adata=adata,
        mark_key=mark_key,
        library_key=library_key,
        spatial_key=spatial_key,
        intensity_key = intensity_key,
        intensity_min = intensity_min,
        intensity_min_key = intensity_min_key,
        calculation_radii=calculation_radii,
        n_interpolated_points=n_interpolated_points,
        marks=marks,
        copy=True,
        compute_labels=True
    )

    sample_gcrossauc_results = np.trapezoid(
        y=sample_gcross_results,
        x=calculation_radii,
        axis=2
    )

    if compute_labels:
        if copy:
            return sample_gcrossauc_results, labels
        else:
            adata.uns[f"{mark_key}_{gcrossauc_inhomogeneous_key}"] = sample_gcrossauc_results
            adata.uns[f"{mark_key}_{gcrossauc_inhomogeneous_key}_labels"] = labels
            return None
    else:
        if copy:
            return sample_gcrossauc_results
        else:
            adata.uns[f"{mark_key}_{gcrossauc_inhomogeneous_key}"] = sample_gcrossauc_results

def gcross(
    point_locations,
    point_marks,
    calculation_radii,
    edge_correction,
    something_edge_matrix,
    n_interpolated_points=100,
    marks = None,
):
    """_summary_

    Args:
        point_locations (_type_): _description_
        point_marks (_type_): _description_
        calculation_radii (_type_): _description_
        edge_correction (_type_): _description_
        something_edge_matrix (_type_): _description_
    """

    point_locations = np.array(point_locations)
    point_marks = np.array(point_marks)
    marks = np.array(marks)

    mark_distances = calculate_mark_distances(
        point_locations=point_locations,
        point_marks=point_marks,
        marks=marks
    )

    edge_distance = calculate_edge_distance(
        point_locations=point_locations,
        n_interpolated_points=n_interpolated_points
    )
    
    # Used to allow marks to be comparable between various patterns
    if marks is None:
        marks = np.sort(np.unique(point_marks))

    if edge_correction is None or edge_correction == "raw":
        return _calculate_gcross_raw(
            mark_distances=mark_distances,
            edge_distance=None,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            marks=marks,
        )
    elif edge_correction == "border":
        return _calculate_gcross_border(
            mark_distances=mark_distances,
            edge_distance=edge_distance,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            marks=marks,
        )
    elif edge_correction == "hanisch":
        area_estimate = ConvexHull(point_locations).volume
        return _calculate_gcross_hanisch(
            mark_distances=mark_distances,
            edge_distance=edge_distance,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            marks=marks,
            area_estimate=area_estimate
        )
    elif edge_correction == "hanisch_biased":
        pass  # TODO maybe just don't implement this
    elif edge_correction == "combined":
        area_estimate = ConvexHull(point_locations).volume
        return _calculate_gcross_combined(
            mark_distances=mark_distances,
            edge_distance=edge_distance,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            marks=marks,
            area_estimate=area_estimate
        )
    elif edge_correction == "kaplan_meier":
        return _calculate_gcross_kaplan_meier(
            mark_distances=mark_distances,
            edge_distance=edge_distance,
            point_marks=point_marks,
            calculation_radii=calculation_radii,
            marks=marks,
        )
    else:
        raise NotImplementedError
    
def calculate_mark_distances(
    point_locations,
    point_marks,
    marks
):
    mark_distances = np.empty((point_locations.shape[0], marks.shape[0]))
    
    for j, mark in enumerate(marks):
        # Must have at least two points of a mark
        if sum(point_marks == mark) >= 2: 
            neigh = NearestNeighbors(n_neighbors=2)
            neigh.fit(point_locations[point_marks == mark, :])
            distances, _ = neigh.kneighbors(point_locations, n_neighbors=2)

            mark_distances[:, j] = distances[:, 0]
            mark_distances[point_marks == mark, j] = distances[point_marks == mark, 1]
        else:
            mark_distances[:, j] = np.nan

    return mark_distances

def calculate_edge_distance(
    point_locations,
    n_interpolated_points=100
):
    pattern_hull = ConvexHull(point_locations)

    x1_ind = pattern_hull.simplices[:, 0]
    x2_ind = pattern_hull.simplices[:, 1]
    edge_directions = point_locations[x2_ind] - point_locations[x1_ind]

    # Add points spaced along simplicies
    hull_points = np.empty((0,2))
    for impute_lambda in np.linspace(0, 1, num = 2 + n_interpolated_points):
        hull_points = np.append(hull_points, edge_directions*impute_lambda + point_locations[x1_ind], axis = 0)

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(hull_points)
    edge_distances, _ = neigh.kneighbors(point_locations, n_neighbors=1)

    return edge_distances

def _calculate_gcross_raw(
    mark_distances,
    edge_distance,
    point_marks,
    calculation_radii,
    marks,
):    
    gcross_results = np.empty((marks.shape[0], marks.shape[0], calculation_radii.shape[0])) # (j, j, r)

    mark_identity_matrix = (point_marks[np.newaxis, :] == marks[:, np.newaxis]).astype(float) # (m, n)
    mark_weights = mark_identity_matrix.sum(axis=1) # (m, )
    mark_weights[mark_weights == 0] = np.inf # Handling divide zero
    for j, _ in enumerate(marks):
        if (~np.isnan(mark_distances[:, [j]])).any():
            nearest_le_calculation_radius = (mark_distances[:, [j]] < calculation_radii[np.newaxis, :]).astype(float) # (n, r)
            gcross_results[:, j, :] = (mark_identity_matrix/mark_weights[:, np.newaxis]) @ nearest_le_calculation_radius # (m, r)
            gcross_results[~mark_identity_matrix.any(axis=1), j, :] = np.nan # No from detected
        else:
            gcross_results[:, j, :] = np.nan  # No marks observed

    return gcross_results

def _calculate_gcross_border(
    mark_distances,
    edge_distance,
    point_marks,
    calculation_radii,
    marks,
):
    gcross_results = np.empty((marks.shape[0], marks.shape[0], calculation_radii.shape[0])) # (j, j, r)

    mark_identity_matrix = (point_marks[np.newaxis, :] == marks[:, np.newaxis]).astype(float) # (m, n)
    edge_ge_calculation_radius = (calculation_radii <= edge_distance).astype(float) # (n, r) 
    edge_ge_sum = mark_identity_matrix @ edge_ge_calculation_radius  # (m, r)

    # Account for divide 0
    edge_ge_sum[edge_ge_sum == 0] = np.inf

    for j, _ in enumerate(marks):
        if (~np.isnan(mark_distances[:, [j]])).any():
            nearest_le_calculation_radius = (mark_distances[:, [j]] < calculation_radii[np.newaxis, :]).astype(float) # (n, r)
            edge_ge_nearest_le_calculation_radius = nearest_le_calculation_radius * edge_ge_calculation_radius # (n, r)
            edge_ge_nearest_le_sum = mark_identity_matrix @ edge_ge_nearest_le_calculation_radius  # (m, r)

            gcross_results[:, j, :] = edge_ge_nearest_le_sum/edge_ge_sum
            # Account for all cells being censored. Is infinite to account for divide 0
            gcross_results[np.isinf(edge_ge_sum[:,-1]), j, :] = np.nan 
        else:
            gcross_results[:, j, :] = np.nan  # No marks observed

    return gcross_results

def _calculate_gcross_hanisch(
    mark_distances,
    edge_distance,
    point_marks,
    calculation_radii,
    marks,
    area_estimate
):
    gcross_results = np.empty((marks.shape[0], marks.shape[0], calculation_radii.shape[0])) # (j, j, r)

    mark_identity_matrix = (point_marks[np.newaxis, :] == marks[:, np.newaxis]).astype(float) # (m, n)

    max_distance_to_edge = edge_distance.max()
    radius_area_estimates = area_estimate*((max_distance_to_edge-calculation_radii)/max_distance_to_edge)**2 # (r,)
    # Account for divide 0
    radius_area_estimates[radius_area_estimates == 0] = np.inf

    for j, _ in enumerate(marks):
        if (~np.isnan(mark_distances[:, [j]])).any():
            nearest_area_estimates = area_estimate*((max_distance_to_edge-mark_distances[:, j])/max_distance_to_edge)**2 # (n,)
            nearest_area_estimates[nearest_area_estimates == 0] = np.inf  # Account for divide 0

            nearest_le_calculation_radius = (mark_distances[:, [j]] < calculation_radii[np.newaxis, :]).astype(float) # (n, r)
            nearest_le_edge = (mark_distances[:, [j]] < edge_distance).astype(float) # (n, 1)

            area_scaled_nearest_le_edge = nearest_le_edge/nearest_area_estimates[:, np.newaxis] # (n, 1)
            area_scaled_nearest_le_calculation_radius_le_edge = nearest_le_calculation_radius * area_scaled_nearest_le_edge # (n, r)
            area_scaled_nearest_le_edge_sum = mark_identity_matrix @ area_scaled_nearest_le_edge  # (m, r)
            area_scaled_nearest_le_edge_sum[area_scaled_nearest_le_edge_sum == 0] = np.inf  # Account for divide 0
            area_scaled_nearest_le_calculation_radius_le_edge_sum = mark_identity_matrix @ area_scaled_nearest_le_calculation_radius_le_edge  # (m, r)

            gcross_results[:, j, :] = area_scaled_nearest_le_calculation_radius_le_edge_sum/area_scaled_nearest_le_edge_sum
            # Account for all reference cells being censored
            gcross_results[np.isinf(area_scaled_nearest_le_edge_sum.flatten()), j, :] = np.nan
        else:
            gcross_results[:, j, :] = np.nan  # No marks observed

    return gcross_results

def _calculate_gcross_combined(
    mark_distances,
    edge_distance,
    point_marks,
    calculation_radii,
    marks,
    area_estimate
):
    gcross_results = np.empty((marks.shape[0], marks.shape[0], calculation_radii.shape[0])) # (j, j, r)

    mark_identity_matrix = (point_marks[np.newaxis, :] == marks[:, np.newaxis]).astype(float) # (m, n)
    edge_ge_calculation_radius = (calculation_radii <= edge_distance).astype(float) # (n, r) 
    edge_ge_sum = mark_identity_matrix @ edge_ge_calculation_radius  # (m, r)
    
    max_distance_to_edge = edge_distance.max()
    for j, _ in enumerate(marks):
        if (~np.isnan(mark_distances[:, [j]])).any():
            nearest_area_estimates = area_estimate*((max_distance_to_edge-mark_distances[:, j])/max_distance_to_edge)**2 # (n,)
            nearest_area_estimates[nearest_area_estimates == 0] = np.inf  # Account for divide 0

            nearest_le_calculation_radius = (mark_distances[:, [j]] < calculation_radii[np.newaxis, :]).astype(float) # (n, r)
            nearest_le_edge = (mark_distances[:, [j]] < edge_distance).astype(float) # (n, 1)

            area_scaled_nearest_le_edge = nearest_le_edge/nearest_area_estimates[:, np.newaxis] # (n, 1)
            area_scaled_nearest_le_edge_sum = mark_identity_matrix @ area_scaled_nearest_le_edge  # (m, r)

            area_scaled_nearest_le_calculation_radius_le_edge = nearest_le_calculation_radius * area_scaled_nearest_le_edge # (n, r)
            area_scaled_nearest_le_calculation_radius_le_edge_sum = mark_identity_matrix @ area_scaled_nearest_le_calculation_radius_le_edge  # (m, r)


            nearest_le_calculation_radius = (mark_distances[:, [j]] < calculation_radii[np.newaxis, :]).astype(float) # (n, r)
            edge_ge_nearest_le_calculation_radius = nearest_le_calculation_radius * edge_ge_calculation_radius # (n, r)
            edge_ge_nearest_le_sum = mark_identity_matrix @ edge_ge_nearest_le_calculation_radius  # (m, r)

            combined_numerator = area_scaled_nearest_le_calculation_radius_le_edge_sum + edge_ge_nearest_le_sum
            combined_denominator = area_scaled_nearest_le_edge_sum + edge_ge_sum
            combined_denominator[combined_denominator == 0] = np.inf  # Account for divide 0

            gcross_results[:, j, :] = combined_numerator/combined_denominator
            # Account for all reference cells being censored
            gcross_results[np.isinf(combined_denominator[:,-1]), j, :] = np.nan
        else:
            gcross_results[:, j, :] = np.nan  # No marks observed

    return gcross_results

def _calculate_gcross_kaplan_meier(
    mark_distances,
    edge_distance,
    point_marks,
    calculation_radii,
    marks,
):
    gcross_results = np.empty((marks.shape[0], marks.shape[0], calculation_radii.shape[0])) # (j, j, r)
    mark_identity_matrix = (point_marks[np.newaxis, :] == marks[:, np.newaxis]).astype(float) # (m, n)

    for j, _ in enumerate(marks):
        if (~np.isnan(mark_distances[:, [j]])).any():
            for i, reference_mark in enumerate(marks):
                mark_to_edge_distances = edge_distance[point_marks == reference_mark]
                mark_to_mark_distances = mark_distances[point_marks == reference_mark, [j]]

                unique_mark_mark_distances, _ = np.unique(mark_to_mark_distances, return_counts=True) # (u, )
                unique_calculation_distances = unique_mark_mark_distances[unique_mark_mark_distances <= calculation_radii.max()] 
                if len(unique_calculation_distances) > 0:
                    if 0 != unique_calculation_distances[0]:
                        unique_calculation_distances = np.concatenate(
                            [
                                [0],
                                unique_calculation_distances.flatten(),
                                [calculation_radii.max()]
                            ]
                        )
                    else:
                        unique_calculation_distances = np.concatenate(
                            [
                                unique_calculation_distances.flatten(),
                                [calculation_radii.max()]
                            ]
                        )

                    unique_calculation_radius_le_edge = mark_to_edge_distances >= unique_calculation_distances[np.newaxis, :]
                    kaplan_meier_numerator =  (
                        unique_calculation_radius_le_edge * (mark_to_mark_distances[:, np.newaxis] == unique_calculation_distances[np.newaxis, :])
                        ).sum(axis=0)
                    kaplan_meier_denominator =  (
                        unique_calculation_radius_le_edge * (mark_to_mark_distances[:, np.newaxis] >= unique_calculation_distances[np.newaxis, :])
                        ).sum(axis=0)
                    kaplan_meier_denominator[kaplan_meier_denominator == 0] = 1

                    kaplan_meier_values = 1 - kaplan_meier_numerator/kaplan_meier_denominator
                    cumulative_kaplan_meier_values = 1 - np.cumprod(kaplan_meier_values)
                    cumulative_kaplan_meier_values[np.isnan(cumulative_kaplan_meier_values)] = 1

                    calculation_unique_distance_mapping = (unique_calculation_distances[:, np.newaxis] <= calculation_radii[np.newaxis, :]).sum(axis=0) - 1

                    gcross_results[i, j, :] = cumulative_kaplan_meier_values[calculation_unique_distance_mapping]
                    gcross_results[~mark_identity_matrix.any(axis=1), j, :] = np.nan # No reference marks observed
                else:
                    gcross_results[i, j, :] = 0  # Highest distance shorter than observed nearest observation
        else:
            gcross_results[:, j, :] = np.nan  # No target marks observed

    return gcross_results