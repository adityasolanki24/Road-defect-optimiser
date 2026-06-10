from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from models import Defect, OptimisedRoute, RoutePoint, Weights
from services.ranking_service import calculate_priority_score, prioritize_defects


AVERAGE_CREW_SPEED_KMH = 32
LABOUR_COST_PER_HOUR = 185
VEHICLE_COST_PER_KM = 1.35
DEPOT_LATITUDE = -37.8136
DEPOT_LONGITUDE = 144.9631
MAX_ROUTE_STOPS = 55


@dataclass(frozen=True)
class Coordinate:
    latitude: float
    longitude: float


class RouteService:
    def __init__(self) -> None:
        self._cache: dict[tuple, OptimisedRoute] = {}

    def compute_route(
        self,
        defects: list[Defect],
        weights: Weights,
        data_version: int = 0,
        max_stops: int = MAX_ROUTE_STOPS,
    ) -> OptimisedRoute:
        cache_key = (
            data_version,
            max_stops,
            round(weights.severity, 4),
            round(weights.traffic_density, 4),
            round(weights.location_importance, 4),
            round(weights.risk_factor, 4),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        route = self._build_route(defects, weights, max_stops)

        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[cache_key] = route
        return route

    def _build_route(
        self,
        defects: list[Defect],
        weights: Weights,
        max_stops: int,
    ) -> OptimisedRoute:
        if not defects:
            return OptimisedRoute(
                route=[],
                total_distance=0,
                total_time=0,
                estimated_cost=0,
            )

        selected = prioritize_defects(defects, weights)[:max_stops]
        coordinates = [Coordinate(DEPOT_LATITUDE, DEPOT_LONGITUDE)] + [
            Coordinate(defect.latitude, defect.longitude) for defect in selected
        ]
        graph = self._build_knn_graph(coordinates, neighbours=8)
        priority_lookup = {
            index + 1: calculate_priority_score(defect, weights)
            for index, defect in enumerate(selected)
        }

        current_node = 0
        remaining_nodes = set(range(1, len(coordinates)))
        ordered_nodes: list[int] = []
        total_distance = 0.0

        while remaining_nodes:
            next_node, segment_distance = self._choose_next_stop(
                current_node,
                remaining_nodes,
                coordinates,
                graph,
                priority_lookup,
            )
            ordered_nodes.append(next_node)
            total_distance += segment_distance
            remaining_nodes.remove(next_node)
            current_node = next_node

        route_points: list[RoutePoint] = []
        repair_hours = 0.0

        for sequence, node_index in enumerate(ordered_nodes, start=1):
            defect = selected[node_index - 1]
            repair_hours += defect.estimated_repair_time_hours
            route_points.append(
                RoutePoint(
                    defect_id=defect.id,
                    latitude=defect.latitude,
                    longitude=defect.longitude,
                    priority_score=priority_lookup[node_index],
                    severity=defect.severity,
                    estimated_repair_time_hours=defect.estimated_repair_time_hours,
                    sequence=sequence,
                )
            )

        travel_hours = total_distance / AVERAGE_CREW_SPEED_KMH
        total_time = repair_hours + travel_hours
        estimated_cost = total_time * LABOUR_COST_PER_HOUR + total_distance * VEHICLE_COST_PER_KM

        return OptimisedRoute(
            route=route_points,
            total_distance=round(total_distance, 2),
            total_time=round(total_time, 2),
            estimated_cost=round(estimated_cost, 2),
        )

    def _choose_next_stop(
        self,
        current_node: int,
        remaining_nodes: set[int],
        coordinates: list[Coordinate],
        graph: dict[int, list[tuple[int, float]]],
        priority_lookup: dict[int, float],
    ) -> tuple[int, float]:
        best_node = -1
        best_distance = 0.0
        best_score = math.inf

        for candidate in remaining_nodes:
            distance = self._a_star_distance(current_node, candidate, coordinates, graph)
            priority = priority_lookup[candidate]
            weighted_cost = distance * (1.0 - min(priority, 1.0) * 0.32)

            if weighted_cost < best_score:
                best_score = weighted_cost
                best_node = candidate
                best_distance = distance

        return best_node, best_distance

    def _build_knn_graph(
        self,
        coordinates: list[Coordinate],
        neighbours: int,
    ) -> dict[int, list[tuple[int, float]]]:
        graph: dict[int, list[tuple[int, float]]] = {index: [] for index in range(len(coordinates))}

        for index, origin in enumerate(coordinates):
            distances = sorted(
                (
                    (
                        other_index,
                        haversine_km(origin, destination),
                    )
                    for other_index, destination in enumerate(coordinates)
                    if other_index != index
                ),
                key=lambda item: item[1],
            )

            for other_index, distance in distances[:neighbours]:
                graph[index].append((other_index, distance))
                graph[other_index].append((index, distance))

        return graph

    def _a_star_distance(
        self,
        start: int,
        goal: int,
        coordinates: list[Coordinate],
        graph: dict[int, list[tuple[int, float]]],
    ) -> float:
        frontier: list[tuple[float, int]] = [(0.0, start)]
        best_cost: dict[int, float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                return best_cost[current]

            for neighbour, segment_cost in graph[current]:
                next_cost = best_cost[current] + segment_cost
                if next_cost < best_cost.get(neighbour, math.inf):
                    best_cost[neighbour] = next_cost
                    heuristic = haversine_km(coordinates[neighbour], coordinates[goal])
                    heapq.heappush(frontier, (next_cost + heuristic, neighbour))

        return haversine_km(coordinates[start], coordinates[goal])


def haversine_km(origin: Coordinate, destination: Coordinate) -> float:
    radius_km = 6371.0
    lat1 = math.radians(origin.latitude)
    lon1 = math.radians(origin.longitude)
    lat2 = math.radians(destination.latitude)
    lon2 = math.radians(destination.longitude)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


route_service = RouteService()
