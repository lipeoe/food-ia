import json
from collections import defaultdict
from datetime import datetime


class ConveyorCounter:
    def __init__(
        self,
        min_seen_frames=3,
        line_ratio=0.5,      # linha no meio da tela
        direction="lr",      # lr = esquerda->direita | rl = direita->esquerda
        cross_margin=12,     # zona morta para evitar jitter na linha
    ):
        self.min_seen_frames = min_seen_frames
        self.line_ratio = line_ratio
        self.direction = direction
        self.cross_margin = cross_margin

        self.counted_ids = set()
        self.totals = defaultdict(int)
        self.history = []
        self.started_at = datetime.now().isoformat()

        # estado por track_id
        self.track_state = {}

    def get_line_x(self, frame_width: int) -> int:
        return int(frame_width * self.line_ratio)

    def _side(self, cx: float, line_x: int) -> str:
        if cx < (line_x - self.cross_margin):
            return "left"
        if cx > (line_x + self.cross_margin):
            return "right"
        return "on_line"

    def update_from_active_tracks(self, active_tracks, frame_shape):
        """Conta quando o ID cruza a linha no sentido configurado (uma única vez)."""
        _, w = frame_shape[:2]
        line_x = self.get_line_x(w)
        newly_counted = []

        for tr in active_tracks:
            tid = tr["id"]
            x1, y1, x2, y2 = tr["bbox"]
            cx = (x1 + x2) / 2.0
            side = self._side(cx, line_x)

            state = self.track_state.setdefault(
                tid,
                {"class": tr["class"], "last_side": side}
            )
            state["class"] = tr["class"]

            if tid not in self.counted_ids and tr.get("seen_frames", 0) >= self.min_seen_frames:
                last_side = state["last_side"]

                crossed = False
                if self.direction == "lr":
                    crossed = (last_side == "left" and side == "right")
                else:  # rl
                    crossed = (last_side == "right" and side == "left")

                if crossed:
                    cls = state["class"]
                    self.counted_ids.add(tid)
                    self.totals[cls] += 1
                    event = {
                        "id": tid,
                        "class": cls,
                        "timestamp": datetime.now().isoformat(),
                    }
                    self.history.append(event)
                    newly_counted.append(event)

            if side != "on_line":
                state["last_side"] = side

        return newly_counted

    def cleanup_finished_tracks(self, finished_tracks):
        for tr in finished_tracks:
            self.track_state.pop(tr["id"], None)

    def get_json_report(self):
        return {
            "started_at": self.started_at,
            "generated_at": datetime.now().isoformat(),
            "total_items": int(sum(self.totals.values())),
            "totals": dict(self.totals),
            "history": self.history,
        }

    def reset(self):
        self.counted_ids.clear()
        self.totals.clear()
        self.history.clear()
        self.track_state.clear()
        self.started_at = datetime.now().isoformat()