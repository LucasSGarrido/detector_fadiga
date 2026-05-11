from src.features import LEFT_EYE, MOUTH, RIGHT_EYE, eye_aspect_ratio, mouth_aspect_ratio


def _points_with(indices):
    points = [(0.0, 0.0)] * 468
    for index, point in indices.items():
        points[index] = point
    return points


def test_eye_aspect_ratio_open_eye():
    points = _points_with(
        {
            LEFT_EYE[0]: (0.0, 0.0),
            LEFT_EYE[1]: (1.0, -1.0),
            LEFT_EYE[2]: (3.0, -1.0),
            LEFT_EYE[3]: (4.0, 0.0),
            LEFT_EYE[4]: (3.0, 1.0),
            LEFT_EYE[5]: (1.0, 1.0),
        }
    )

    assert eye_aspect_ratio(points, LEFT_EYE) == 0.5


def test_mouth_aspect_ratio():
    points = _points_with(
        {
            MOUTH[0]: (0.0, 0.0),
            MOUTH[1]: (2.0, -1.0),
            MOUTH[2]: (4.0, 0.0),
            MOUTH[3]: (2.0, 1.0),
        }
    )

    assert mouth_aspect_ratio(points) == 0.5


def test_eye_aspect_ratio_works_for_right_eye_indices():
    points = _points_with(
        {
            RIGHT_EYE[0]: (0.0, 0.0),
            RIGHT_EYE[1]: (1.0, -0.5),
            RIGHT_EYE[2]: (3.0, -0.5),
            RIGHT_EYE[3]: (4.0, 0.0),
            RIGHT_EYE[4]: (3.0, 0.5),
            RIGHT_EYE[5]: (1.0, 0.5),
        }
    )

    assert eye_aspect_ratio(points, RIGHT_EYE) == 0.25
