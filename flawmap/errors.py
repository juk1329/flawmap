"""Failures with a code, a Korean explanation and a hint for the person who filmed the device.

The pipeline raises these where a capture cannot be processed for a reason the seller can fix
(device not found, no camera poses); the upload service shows the message and the hint as they are.
"""
from __future__ import annotations

MESSAGES: dict[str, tuple[str, str]] = {
    "UPLOAD_INVALID": ("올린 파일을 열 수 없습니다.",
                       "Stray Scanner 폴더를 파일 앱에서 '압축'해 만든 zip 파일을 올려 주세요."),
    "NOT_STRAY": ("Stray Scanner 촬영본이 아닙니다.",
                  "zip 안에 rgb.mp4, odometry.csv, camera_matrix.csv가 있어야 합니다. 앱에서 내보낸 폴더를 그대로 압축해 주세요."),
    "VIDEO_SHORT": ("촬영이 너무 짧습니다.",
                    "기기 둘레를 낮은 각도, 중간, 높은 각도로 한 바퀴씩, 면마다 60–90초 동안 천천히 찍어 주세요."),
    "NO_POSE": ("촬영한 카메라의 위치를 알 수 없습니다.",
                "Stray Scanner로 찍은 원본 폴더를 올려 주세요. 촬영 중에 화면을 가리거나 너무 빨리 움직이면 위치가 끊깁니다."),
    "DEVICE_NOT_FOUND": ("영상에서 기기의 외곽을 맞추지 못했습니다.",
                         "기기와 색이 다른 바닥 위에 기기를 놓고, 기기가 화면의 30–60%를 차지하게 찍어 주세요. 고른 조명이 좋습니다."),
    "LOW_COVERAGE": ("찍히지 않은 부위가 많습니다.",
                     "안내에 나온 부위를 더 찍어 다시 올리면 그 부위까지 확인할 수 있습니다."),
    "DISK_FULL": ("서버 저장 공간이 부족해 지금은 받을 수 없습니다.", "잠시 뒤에 다시 올려 주세요."),
    "INTERNAL": ("처리 중에 서버 오류가 났습니다.", "같은 촬영본으로 다시 시도해 주세요. 계속되면 운영자에게 알려 주세요."),
}


class FlawmapError(Exception):
    """An error with a code from ``MESSAGES`` and a technical ``detail`` for the logs."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code if code in MESSAGES else "INTERNAL"
        self.detail = detail
        super().__init__(f"{self.code}: {detail}" if detail else self.code)

    @property
    def message(self) -> str:
        return MESSAGES[self.code][0]

    @property
    def hint(self) -> str:
        return MESSAGES[self.code][1]

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint, "detail": self.detail}
