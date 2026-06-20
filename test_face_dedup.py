import math
import unittest

import numpy as np

from face_dedup import (
    best_reference_match,
    cluster_face_indices,
    deduplicate_detection_indices,
    flatten_references,
    match_face_indices_iteratively,
)


def unit_vector(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)


class FaceDedupTests(unittest.TestCase):
    def test_unmatched_similar_faces_are_clustered(self):
        vectors = [unit_vector(0), unit_vector(15), unit_vector(90)]

        clusters = cluster_face_indices(
            vectors,
            [0, 1, 2],
            similarity_threshold=0.90,
        )

        self.assertEqual([[0, 1], [2]], clusters)

    def test_clear_face_bridges_hard_face_to_existing_person(self):
        existing = unit_vector(0)
        easy_face = unit_vector(50)
        hard_face = unit_vector(85)
        vectors = [easy_face, hard_face]

        assignments, remaining, _, passes = match_face_indices_iteratively(
            vectors,
            [0, 1],
            {"person-1": [existing]},
            similarity_threshold=0.58,
        )

        self.assertEqual({0: "person-1", 1: "person-1"}, assignments)
        self.assertEqual([], remaining)
        self.assertEqual(2, passes)
        self.assertLess(float(hard_face @ existing), 0.58)

    def test_representative_face_can_match_when_centroid_does_not(self):
        query = unit_vector(0)
        stale_centroid = unit_vector(65)
        representative = unit_vector(20)
        references, labels = flatten_references(
            {"person-1": [stale_centroid, representative]}
        )

        person_id, score = best_reference_match([query], references, labels)

        self.assertEqual("person-1", person_id)
        self.assertGreaterEqual(score, 0.58)
        self.assertLess(float(query @ stale_centroid), 0.58)

    def test_overlapping_face_detections_keep_highest_confidence(self):
        kept = deduplicate_detection_indices(
            bboxes=[
                [10, 10, 110, 110],
                [12, 12, 108, 108],
                [150, 20, 220, 90],
            ],
            scores=[0.91, 0.82, 0.80],
            iou_threshold=0.65,
        )

        self.assertEqual([0, 2], kept)


if __name__ == "__main__":
    unittest.main()
