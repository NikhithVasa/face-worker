from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def cluster_face_indices(
    vectors: Sequence[np.ndarray],
    indices: Sequence[int],
    similarity_threshold: float,
    block_size: int = 512,
) -> List[List[int]]:
    """Cluster selected face vectors and return groups of original indices."""
    selected = list(indices)
    if not selected:
        return []
    if len(selected) == 1:
        return [selected]

    matrix = np.stack([vectors[i] for i in selected]).astype(np.float32)
    parents = list(range(len(selected)))
    ranks = [0] * len(selected)

    def find(item: int) -> int:
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if ranks[root_a] < ranks[root_b]:
            root_a, root_b = root_b, root_a
        parents[root_b] = root_a
        if ranks[root_a] == ranks[root_b]:
            ranks[root_a] += 1

    for start in range(0, len(selected), block_size):
        end = min(len(selected), start + block_size)
        similarities = matrix[start:end] @ matrix.T
        for local_i in range(end - start):
            i = start + local_i
            for j in range(i + 1, len(selected)):
                if float(similarities[local_i, j]) >= similarity_threshold:
                    union(i, j)

    groups: Dict[int, List[int]] = {}
    for local_i, original_i in enumerate(selected):
        groups.setdefault(find(local_i), []).append(original_i)
    return list(groups.values())


def flatten_references(
    references_by_identity: Mapping[Hashable, Sequence[np.ndarray]],
) -> Tuple[Optional[np.ndarray], List[Hashable]]:
    vectors: List[np.ndarray] = []
    labels: List[Hashable] = []
    for identity, identity_vectors in references_by_identity.items():
        for vector in identity_vectors:
            vectors.append(np.asarray(vector, dtype=np.float32))
            labels.append(identity)

    if not vectors:
        return None, []
    return np.stack(vectors).astype(np.float32), labels


def best_reference_match(
    query_vectors: Sequence[np.ndarray],
    reference_matrix: Optional[np.ndarray],
    reference_labels: Sequence[Hashable],
) -> Tuple[Optional[Hashable], float]:
    """Return the identity of the strongest query/reference pair."""
    if not query_vectors or reference_matrix is None or not reference_labels:
        return None, -1.0

    queries = np.stack(query_vectors).astype(np.float32)
    similarities = queries @ reference_matrix.T
    flat_index = int(np.argmax(similarities))
    _, reference_index = np.unravel_index(flat_index, similarities.shape)
    return reference_labels[reference_index], float(similarities.flat[flat_index])


def match_face_indices_iteratively(
    vectors: Sequence[np.ndarray],
    indices: Sequence[int],
    references_by_identity: Mapping[Hashable, Sequence[np.ndarray]],
    similarity_threshold: float,
    batch_size: int = 2048,
) -> Tuple[Dict[int, Hashable], List[int], Dict[Hashable, List[np.ndarray]], int]:
    """
    Match faces repeatedly, adding successful matches as new references.

    This lets a clear face bridge a harder pose to an existing identity without
    requiring all album faces to be clustered first.
    """
    references: Dict[Hashable, List[np.ndarray]] = {
        identity: [np.asarray(vector, dtype=np.float32) for vector in identity_vectors]
        for identity, identity_vectors in references_by_identity.items()
    }
    assignments: Dict[int, Hashable] = {}
    remaining = list(indices)
    passes = 0

    while remaining and references:
        reference_matrix, reference_labels = flatten_references(references)
        if reference_matrix is None:
            break

        still_remaining: List[int] = []
        matched_vectors: Dict[Hashable, List[np.ndarray]] = {}

        for start in range(0, len(remaining), batch_size):
            batch_indices = remaining[start:start + batch_size]
            queries = np.stack([vectors[i] for i in batch_indices]).astype(np.float32)
            similarities = queries @ reference_matrix.T
            best_reference_indices = similarities.argmax(axis=1)
            best_scores = similarities.max(axis=1)

            for local_i, score in enumerate(best_scores):
                face_index = batch_indices[local_i]
                if float(score) < similarity_threshold:
                    still_remaining.append(face_index)
                    continue

                identity = reference_labels[int(best_reference_indices[local_i])]
                assignments[face_index] = identity
                matched_vectors.setdefault(identity, []).append(
                    np.asarray(vectors[face_index], dtype=np.float32)
                )

        if not matched_vectors:
            break

        for identity, identity_vectors in matched_vectors.items():
            references.setdefault(identity, []).extend(identity_vectors)

        remaining = still_remaining
        passes += 1

    return assignments, remaining, references, passes


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def deduplicate_detection_indices(
    bboxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> List[int]:
    """Keep the highest-confidence detection from each heavily-overlapping set."""
    order = sorted(range(len(bboxes)), key=lambda i: float(scores[i]), reverse=True)
    kept: List[int] = []
    for index in order:
        if all(bbox_iou(bboxes[index], bboxes[other]) < iou_threshold for other in kept):
            kept.append(index)
    return kept
