"""Main matching logic controller."""

from ..utils.logger import get_logger
from ..utils.helpers import _listify_ids
from ..config import (
    DEFAULT_ENDPOINT_GRID_M, DEFAULT_MATCH_BUFFER_M, DEFAULT_MIN_COVERAGE,
    DEFAULT_TARGET_RATIO, DEFAULT_OVERLAP_IOU
)


logger = get_logger(__name__)


class GridMatcher:
    """
    Main matcher class that orchestrates the matching process.

    Parameters
    ----------
    jao_gdf : GeoDataFrame
        JAO transmission line data
    pypsa_gdf : GeoDataFrame
        PyPSA transmission line data
    endpoint_grid_m : float, optional
        Grid size in meters for endpoint snapping
    match_buffer_m : float, optional
        Buffer size in meters for corridor matching
    min_coverage : float, optional
        Minimum coverage ratio for matching
    target_ratio : float, optional
        Target ratio for extending short matches
    """

    def __init__(self, jao_gdf, pypsa_gdf,
                 endpoint_grid_m=DEFAULT_ENDPOINT_GRID_M,
                 match_buffer_m=DEFAULT_MATCH_BUFFER_M,
                 min_coverage=DEFAULT_MIN_COVERAGE,
                 target_ratio=DEFAULT_TARGET_RATIO,
                 overlap_iou=DEFAULT_OVERLAP_IOU):
        self.jao_gdf = jao_gdf
        self.pypsa_gdf = pypsa_gdf
        self.endpoint_grid_m = endpoint_grid_m
        self.match_buffer_m = match_buffer_m
        self.min_coverage = min_coverage
        self.target_ratio = target_ratio
        self.overlap_iou = overlap_iou

        # Will be populated during matching process
        self.results = []
        self.matched_jao_ids = set()
        self.never_match_jao_ids = set()
        self.network_graph = None
        self.chains = None
        self.segment_to_chain = None

    def run_matching(self):
        """
        Run the complete matching process.

        Returns
        -------
        list
            List of match dictionaries
        """
        logger.info("Starting the matching process")

        # Stage 1: Build network graph and chains
        from ..geo.network import build_network_graph
        from .chain import build_pypsa_chains

        self.network_graph = build_network_graph(self.pypsa_gdf)
        self.chains, self.segment_to_chain = build_pypsa_chains(
            self.pypsa_gdf, endpoint_grid_m=self.endpoint_grid_m
        )

        # Stage 2: Match known corridors
        self._match_known_corridors()

        # Stage 3: Chain corridor matching
        self._match_chain_corridors()

        # Stage 4: Bus-based path matching
        self._match_bus_paths()

        # Stage 5: Path-based matching
        self._match_line_paths()

        # Stage 6: Fix parallel circuits
        self._fix_parallel_circuits()

        # Stage 7: Auto-extend short matches
        self._extend_short_matches()

        # Stage 8: Circuit-aware matching
        self._apply_circuit_aware_matching()

        # Stage 9: Allocate electrical parameters
        self._allocate_parameters()

        # Final enforcement for forced unmatched lines
        self._enforce_unmatched_constraints()

        logger.info(f"Matching completed: {len(self.matched_jao_ids)} of {len(self.jao_gdf)} JAO lines matched")

        return self.results

    def _match_known_corridors(self):
        """Match known problem corridors using exact mappings."""
        from .corridor import match_multi_segment_corridors

        corridor_matches, forced_unmatched = match_multi_segment_corridors(
            self.jao_gdf, self.pypsa_gdf
        )

        self.results.extend(corridor_matches)
        self.never_match_jao_ids.update(forced_unmatched)
        self.matched_jao_ids.update(
            str(m['jao_id']) for m in corridor_matches if m.get('matched', False)
        )

        logger.info(
            f"Matched {sum(1 for m in corridor_matches if m.get('matched', False))} JAO lines with corridor matching")
        logger.info(f"Added {len(forced_unmatched)} JAO IDs to never-match list")

    def _match_chain_corridors(self):
        """Match unmatched JAO lines using chain corridors."""
        from .chain import chain_corridor_rematch

        # Only apply to unmatched JAO lines that aren't in never_match_jao_ids
        unmatched_jao_gdf = self.jao_gdf[
            (~self.jao_gdf['id'].astype(str).isin(self.matched_jao_ids)) &
            (~self.jao_gdf['id'].astype(str).isin(self.never_match_jao_ids))
            ]

        if len(unmatched_jao_gdf) > 0:
            chain_matches = chain_corridor_rematch(
                pypsa_gdf=self.pypsa_gdf,
                jao_matches=unmatched_jao_gdf,
                endpoint_grid_m=self.endpoint_grid_m,
                match_buffer_m=self.match_buffer_m,
                min_coverage=self.min_coverage,
                verbose=True,
                existing_results=self.results
            )

            # Add new matches to results
            for m in chain_matches:
                jao_id = str(m['jao_id'])
                if m.get('matched',
                         False) and jao_id not in self.matched_jao_ids and jao_id not in self.never_match_jao_ids:
                    self.results.append(m)
                    self.matched_jao_ids.add(jao_id)

            logger.info(
                f"Matched {sum(1 for m in chain_matches if m.get('matched', False) and str(m['jao_id']) not in self.matched_jao_ids)} additional JAO lines with chain corridor matching")

    def _match_bus_paths(self):
        """Match unmatched JAO lines using bus-based paths."""
        from .paths import find_bus_based_paths

        # Only apply to unmatched JAO lines that aren't in never_match_jao_ids
        unmatched_jao_gdf = self.jao_gdf[
            (~self.jao_gdf['id'].astype(str).isin(self.matched_jao_ids)) &
            (~self.jao_gdf['id'].astype(str).isin(self.never_match_jao_ids))
            ]

        if len(unmatched_jao_gdf) > 0:
            bus_matches = find_bus_based_paths(unmatched_jao_gdf, self.pypsa_gdf, self.network_graph)

            # Add new matches to results
            for m in bus_matches:
                jao_id = str(m['jao_id'])
                if m.get('matched',
                         False) and jao_id not in self.matched_jao_ids and jao_id not in self.never_match_jao_ids:
                    self.results.append(m)
                    self.matched_jao_ids.add(jao_id)

            logger.info(
                f"Matched {sum(1 for m in bus_matches if m.get('matched', False))} JAO lines with bus-based path matching")

    def _match_line_paths(self):
        """Match unmatched JAO lines using path-based matching."""
        from grid_matcher.geo.network import find_nearest_endpoints
        from .paths import path_based_line_matching
        from ..geo.network import identify_duplicate_geometries
        from collections import defaultdict

        # Only apply to unmatched JAO lines that aren't in never_match_jao_ids
        unmatched_jao_gdf = self.jao_gdf[
            (~self.jao_gdf['id'].astype(str).isin(self.matched_jao_ids)) &
            (~self.jao_gdf['id'].astype(str).isin(self.never_match_jao_ids))
            ]

        if len(unmatched_jao_gdf) > 0:
            # Identify parallel groups
            parallel_groups, jao_to_group = identify_duplicate_geometries(self.jao_gdf)

            # Find nearest endpoints and perform path-based matching
            nearest_points = find_nearest_endpoints(unmatched_jao_gdf, self.pypsa_gdf, self.network_graph)
            path_matches = path_based_line_matching(
                unmatched_jao_gdf, self.pypsa_gdf, self.network_graph,
                nearest_points, parallel_groups, jao_to_group,
                existing_matches=self.results
            )

            # Add new matches to results
            for m in path_matches:
                jao_id = str(m['jao_id'])
                if m.get('matched',
                         False) and jao_id not in self.matched_jao_ids and jao_id not in self.never_match_jao_ids:
                    self.results.append(m)
                    self.matched_jao_ids.add(jao_id)

            logger.info(
                f"Matched {sum(1 for m in path_matches if m.get('matched', False))} JAO lines with path-based matching")

    def _fix_parallel_circuits(self):
        """Fix parallel circuit matching."""
        from .parallel import fix_parallel_circuit_matching

        self.results = fix_parallel_circuit_matching(self.results, self.jao_gdf, self.pypsa_gdf)

        # Update matched IDs
        self.matched_jao_ids = set(
            str(m['jao_id']) for m in self.results if
            m.get('matched', False) and str(m['jao_id']) not in self.never_match_jao_ids
        )

        logger.info("Applied parallel circuit fixes")

    def _extend_short_matches(self):
        """Extend short matches to improve coverage."""
        from .chain import auto_extend_short_matches

        self.results = auto_extend_short_matches(
            self.results, self.jao_gdf, self.pypsa_gdf,
            target_ratio=self.target_ratio, buffer_m=self.match_buffer_m
        )

        logger.info("Extended short matches to improve coverage")

    def _apply_circuit_aware_matching(self):
        """Apply circuit-aware matching."""
        from .parallel import circuit_aware_matching

        self.results = circuit_aware_matching(self.results, self.jao_gdf, self.pypsa_gdf)

        logger.info("Applied circuit-aware matching")

    def _allocate_parameters(self):
        """Allocate electrical parameters."""
        from .parameter import allocate_electrical_parameters

        self.results = allocate_electrical_parameters(self.jao_gdf, self.pypsa_gdf, self.results)

        logger.info("Allocated electrical parameters")

    def _enforce_unmatched_constraints(self):
        """Enforce that certain JAO IDs remain unmatched."""
        for i, match in enumerate(self.results):
            jao_id = str(match.get('jao_id', ''))
            if jao_id in self.never_match_jao_ids or match.get('forced_unmatched', False):
                logger.info(f"Enforcing unmatched status for JAO {jao_id}")
                self.results[i]['matched'] = False
                self.results[i]['pypsa_ids'] = []
                self.results[i]['locked_by_corridor'] = True
                self.results[i]['forced_unmatched'] = True

                # Remove from matched IDs if present
                if jao_id in self.matched_jao_ids:
                    self.matched_jao_ids.remove(jao_id)