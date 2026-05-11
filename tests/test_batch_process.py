from pathlib import Path

from batch_process import build_command, iter_videos


def test_iter_videos_filters_known_extensions(tmp_path):
    (tmp_path / "a.mp4").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.MOV").write_text("", encoding="utf-8")

    videos = list(iter_videos(tmp_path, [".mp4", ".mov"]))

    assert videos == [tmp_path / "a.mp4", nested / "c.MOV"]


def test_build_command_uses_video_timestamp_and_headless_mode():
    command = build_command(
        Path("video.mp4"),
        config=Path("config.yaml"),
        save_video=True,
        max_frames=10,
    )

    assert "--headless" in command
    assert "--no-sound" in command
    assert "--timestamp-mode" in command
    assert "video" in command
    assert "--save-video" in command
    assert "--max-frames" in command
