"""Local persistence with evaluator labels stored outside runtime session records."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import uuid

from .image_identity import DECODER_ID, decode_color, rgb_hash


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    paths = [path for path in root.rglob("*") if path.is_file()
             and path.suffix in (".py", ".json") and "__pycache__" not in path.parts]
    paths += list(root.parent.glob("requirements*.txt"))
    for path in sorted(paths):
        digest.update(path.relative_to(root.parent).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_hash(path: Path) -> str:
    return rgb_hash(decode_color(path.read_bytes()))


def artifact_hash(directory: Path) -> str:
    return stable_hash({p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()})


class Storage:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ("datasets", "maps", "runs", "models", "evaluator", "uploads"):
            (self.root / subdir).mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.path = self.root / "state.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "version": 1, "datasets": {}, "sessions": {}, "maps": {},
            "candidates": {}, "runs": {}, "models": {}, "audits": {}, "jobs": {},
        }
        for job in self.state["jobs"].values():
            if job["status"] in ("queued", "running"):
                job.update(status="failed", error="The local server restarted before this job completed.")
        self.state.setdefault("comparisons", {})
        # Decoder changes invalidate legacy or previously migrated pixel identities.
        refresh_pixels = self.state.get("pixel_hash_decoder") != DECODER_ID
        for session in self.state["sessions"].values():
            prefix = session["dataset_id"] + "--"
            session.setdefault("original_group_id", session["group_id"][len(prefix):] if session["group_id"].startswith(prefix) else session["group_id"])
            for frame in session["frames"]:
                if refresh_pixels or "content_hash" not in frame:
                    frame["content_hash"] = pixel_hash(Path(frame["image_path"]))
        self.state["pixel_hash_decoder"] = DECODER_ID
        self.persist()

    def persist(self):
        with self.lock:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(self.path)

    def install_dataset(self, parsed: dict):
        with self.lock:
            dataset_id = parsed["dataset_id"]
            if dataset_id in self.state["datasets"]:
                return False
            self.validate_dataset_provenance(parsed)
            for session in parsed["sessions"]:
                self.state["sessions"][session["id"]] = deepcopy(session)
                labels = parsed.get("labels", {}).get(session["id"])
                if labels:
                    self.labels_path(session["id"]).write_text(json.dumps(labels, allow_nan=False), encoding="utf-8")
            self.state["datasets"][dataset_id] = {
                "id": dataset_id, "name": parsed["name"], "source": parsed["source"],
                "session_ids": [s["id"] for s in parsed["sessions"]],
                "warnings": parsed.get("warnings", []), "created_at": now(),
                "benchmark": deepcopy(parsed.get("benchmark", {})),
                "evaluation": deepcopy(parsed.get("evaluation", {})),
            }
            self.persist()
            return True

    def validate_dataset_provenance(self, parsed: dict):
        content_roles = {}
        group_roles = {}
        for session in self.state["sessions"].values():
            group = session.get("original_group_id", session["group_id"])
            group_roles.setdefault(group, set()).add(session["split"])
            for frame in session["frames"]:
                digest = frame.get("content_hash") or pixel_hash(Path(frame["image_path"]))
                content_roles.setdefault(digest, set()).add(session["split"])
        for session in parsed["sessions"]:
            role = session["split"]
            group = session.get("original_group_id", session["group_id"])
            if group in group_roles and group_roles[group] != {role}:
                raise ValueError("This original recording group already belongs to a different split in the workspace. Renaming a package does not reset its provenance.")
            for frame in session["frames"]:
                digest = frame.get("content_hash") or pixel_hash(Path(frame["image_path"]))
                existing = content_roles.get(digest, set())
                if existing and (existing != {role} or role == "audit"):
                    raise ValueError("These decoded image pixels are already assigned to another split or a registered final benchmark. Use genuinely disjoint recordings.")

    def labels_path(self, session_id: str) -> Path:
        # IDs are looked up in state before use; encoded names cannot traverse paths.
        safe = hashlib.sha256(session_id.encode()).hexdigest()
        return self.root / "evaluator" / f"{safe}.json"

    def get_labels(self, session_id: str):
        if session_id not in self.state["sessions"]:
            raise KeyError(session_id)
        path = self.labels_path(session_id)
        return json.loads(path.read_text()) if path.exists() else None

    def get(self, kind: str, identifier: str) -> dict:
        with self.lock:
            if identifier not in self.state[kind]:
                raise KeyError(identifier)
            return deepcopy(self.state[kind][identifier])

    def values(self, kind: str) -> list[dict]:
        with self.lock:
            return deepcopy(list(self.state[kind].values()))

    def put(self, kind: str, value: dict):
        with self.lock:
            self.state[kind][value["id"]] = deepcopy(value)
            self.persist()

    def save_run(self, run: dict):
        path = self.root / "runs" / f"{run['id']}.json"
        path.write_text(json.dumps(run, allow_nan=False), encoding="utf-8")
        summary = {k: v for k, v in run.items() if k not in ("predictions", "events")}
        if summary.get("metrics"):
            summary["metrics"] = {k: v for k, v in summary["metrics"].items() if k != "per_frame"}
        self.put("runs", summary)

    def load_run(self, identifier: str) -> dict:
        self.get("runs", identifier)
        return json.loads((self.root / "runs" / f"{identifier}.json").read_text())


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"
