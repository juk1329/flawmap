"""Reconstruction candidates for a photo-real, freely rotatable model, compared side by side.

The user's request (2026-11-18): several candidate ways of turning one capture into a realistic 3D
model (the fitted plate with measured textures, image-based blending of real frames, Gaussian
splatting with and without geometry constraints, an RGB-D scan mesh, ...), each built as a real
pipeline, each producing its model, and a shared way to compare them, by numbers and by eye, so
the best one can be chosen for the service and the search can go on.

Every candidate gets the same inputs and is judged the same way:

- ``context.CaptureContext``: one recording (one capture, its own unknown environment), its held-out
  time blocks (``eval.heldout``), the camera of every frame in the device frame (mm, OpenCV axes)
  from a finished inspection run, the fitted device model, and the defects that run found.
- ``registry``: the candidates (``candidates/c*.py``), each with ``fit`` -> artifact, ``render``
  (a view -> RGB + alpha) and ``export_web`` (files + a manifest for the comparison page).
- ``stress``: views a person would rotate to (360-degree turns at several heights, grazing, the
  unfilmed side, close-ups), where Gaussian splatting broke for the user.
- ``metrics``: fidelity on held-out frames, robustness on stress views, practicality.
- ``runner``: ``flawmap compare`` builds the context once, runs the candidates and writes
  ``runs/compare/<recording>/`` (index for ``web/compare.html``, renders, metrics, leaderboard).
"""
