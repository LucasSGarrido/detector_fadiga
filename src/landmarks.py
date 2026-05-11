from __future__ import annotations

from dataclasses import dataclass

from .features import Point


@dataclass(frozen=True)
class FaceDetection:
    points: list[Point] | None

    @property
    def found(self) -> bool:
        return bool(self.points)


class MediaPipeFaceMeshDetector:
    """Wrapper around MediaPipe Face Mesh.

    The project plan mentions Face Landmarker. For the first MVP, Face Mesh is
    used because it runs without downloading an external model asset.
    """

    def __init__(
        self,
        *,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr) -> FaceDetection:
        import cv2

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = self._face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return FaceDetection(points=None)

        face = result.multi_face_landmarks[0]
        points = [(landmark.x * width, landmark.y * height) for landmark in face.landmark]
        return FaceDetection(points=points)

    def close(self) -> None:
        self._face_mesh.close()

    def __enter__(self) -> "MediaPipeFaceMeshDetector":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
