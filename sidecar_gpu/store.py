"""SQLite is the cross-process coordinator; no in-memory job queue is authoritative."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid
from .arrays import file_hash, inspect_object
from .errors import SidecarError

ACTIVE = ("queued", "running", "cancel_requested")
TERMINAL = ("succeeded", "failed", "cancelled", "interrupted")


def identifier(prefix):
    return prefix + uuid.uuid4().hex


class Store:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.data_dir)
        self.blobs = self.root / "objects"
        self.temp = self.root / "tmp"
        self.blobs.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "metadata.sqlite3"

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def initialize(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS objects (
              id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, bytes INTEGER NOT NULL,
              metadata TEXT NOT NULL, created REAL NOT NULL, accessed REAL NOT NULL,
              protected_until REAL NOT NULL, available INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, idem TEXT UNIQUE NOT NULL, fingerprint TEXT NOT NULL,
              params TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
              updated REAL NOT NULL, started REAL, finished REAL, queued_mono REAL NOT NULL,
              recovered INTEGER NOT NULL DEFAULT 0, result TEXT, error TEXT, timings TEXT NOT NULL DEFAULT '{}');
            CREATE TABLE IF NOT EXISTS job_inputs (
              job_id TEXT NOT NULL REFERENCES jobs(id), object_id TEXT NOT NULL REFERENCES objects(id),
              PRIMARY KEY(job_id,object_id));
            CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS leases (id TEXT PRIMARY KEY, object_id TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS runtime (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS job_queue ON jobs(state,created);
            """)

    def path(self, sha256):
        # Hashes originate only from a hasher or validated database entries.
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise SidecarError("invalid_hash", "expected a SHA-256 hex digest", 422)
        return self.blobs / (sha256 + ".npy")

    def _object(self, db, handle):
        row = db.execute("SELECT * FROM objects WHERE id=?", (handle,)).fetchone()
        if row is None:
            raise SidecarError("object_not_found", "unknown object handle", 404)
        if not row["available"] or not self.path(row["sha256"]).is_file():
            raise SidecarError("object_expired", "object has been evicted or is missing; upload it again", 410)
        return row

    @staticmethod
    def object_json(row):
        return {"handle": row["id"], "sha256": row["sha256"], "bytes": row["bytes"],
                "metadata": json.loads(row["metadata"]), "created_at": row["created"],
                "last_access_at": row["accessed"], "protected_until": row["protected_until"]}

    def object(self, handle):
        with self.transaction() as db:
            row = self._object(db, handle)
            db.execute("UPDATE objects SET accessed=? WHERE id=?", (time.time(), handle))
            return self.object_json(row)

    def lookup(self, digest):
        self.path(digest)
        with self.transaction() as db:
            row = db.execute("SELECT * FROM objects WHERE sha256=? AND available=1", (digest,)).fetchone()
            if row and self.path(digest).exists():
                db.execute("INSERT INTO counters VALUES ('object_lookup_hits',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
                now = time.time()
                db.execute("UPDATE objects SET accessed=?,protected_until=MAX(protected_until,?) WHERE id=?",
                           (now, now + self.config.input_grace_seconds, row["id"]))
                return self.object_json(row)
            db.execute("INSERT INTO counters VALUES ('object_lookup_misses',1) ON CONFLICT(name) DO UPDATE SET value=value+1")
        return None

    def _room(self, db, needed, protected=()):
        now = time.time()
        db.execute("DELETE FROM leases WHERE expires<?", (now,))
        used = db.execute("SELECT COALESCE(SUM(bytes),0) FROM objects WHERE available=1").fetchone()[0]
        reserved = db.execute("SELECT COALESCE(SUM(bytes),0) FROM reservations").fetchone()[0]
        if used + reserved + needed <= self.config.disk_budget:
            return True
        candidates = db.execute("""SELECT * FROM objects o WHERE available=1 AND protected_until<=?
          AND NOT EXISTS (SELECT 1 FROM job_inputs i JOIN jobs j ON j.id=i.job_id
            WHERE i.object_id=o.id AND j.state IN ('queued','running','cancel_requested'))
          AND NOT EXISTS (SELECT 1 FROM leases l WHERE l.object_id=o.id)
          ORDER BY accessed ASC""", (now,)).fetchall()
        for row in candidates:
            if row["id"] in protected:
                continue
            try:
                self.path(row["sha256"]).unlink(missing_ok=True)
            except OSError:
                continue
            db.execute("UPDATE objects SET available=0 WHERE id=?", (row["id"],))
            used -= row["bytes"]
            if used + reserved + needed <= self.config.disk_budget:
                return True
        return False

    def reserve(self, key, size):
        with self.transaction() as db:
            fits = self._room(db, size)
            if fits:
                db.execute("INSERT INTO reservations VALUES (?,?)", (key, size))
        if not fits:
            raise SidecarError("storage_full", "disk budget is reserved or protected; retry later or increase it", 507)

    def release(self, key):
        with self.transaction() as db:
            db.execute("DELETE FROM reservations WHERE id=?", (key,))

    def _commit_object(self, db, temporary, digest, metadata, protection, reservation):
        size, now = temporary.stat().st_size, time.time()
        budget = db.execute("SELECT bytes FROM reservations WHERE id=?", (reservation,)).fetchone()
        if budget is None or size > budget[0]:
            raise SidecarError("storage_reservation", "object exceeds its reserved size", 507)
        existing = db.execute("SELECT * FROM objects WHERE sha256=?", (digest,)).fetchone()
        hit = bool(existing and existing["available"] and self.path(digest).exists())
        handle = existing["id"] if existing else identifier("o_")
        if hit:
            temporary.unlink(missing_ok=True)
            db.execute("UPDATE objects SET accessed=?,protected_until=MAX(protected_until,?) WHERE id=?",
                       (now, now + protection, handle))
        else:
            temporary.replace(self.path(digest))
            if existing:
                db.execute("UPDATE objects SET available=1,accessed=?,protected_until=? WHERE id=?",
                           (now, now + protection, handle))
            else:
                db.execute("INSERT INTO objects VALUES (?,?,?,?,?,?,?,1)",
                           (handle, digest, size, json.dumps(metadata), now, now, now + protection))
        db.execute("DELETE FROM reservations WHERE id=?", (reservation,))
        row = db.execute("SELECT * FROM objects WHERE id=?", (handle,)).fetchone()
        return {**self.object_json(row), "cache_hit": hit}

    def commit_upload(self, temporary, digest, metadata, reservation):
        with self.transaction() as db:
            size = temporary.stat().st_size
            obj = self._commit_object(db, temporary, digest, metadata, self.config.input_grace_seconds, reservation)
            db.execute("INSERT INTO counters VALUES ('received_upload_bytes',?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value", (size,))
            return obj

    def lease(self, handle):
        with self.transaction() as db:
            row = self._object(db, handle)
            lease = identifier("l_")
            db.execute("INSERT INTO leases VALUES (?,?,?)", (lease, handle, time.time() + 3600))
            db.execute("UPDATE objects SET accessed=? WHERE id=?", (time.time(), handle))
            return self.path(row["sha256"]), lease

    def release_lease(self, lease):
        with self.transaction() as db:
            db.execute("DELETE FROM leases WHERE id=?", (lease,))

    @staticmethod
    def job_json(row):
        return {"id": row["id"], "idempotency_key": row["idem"], "operation": json.loads(row["params"])["operation"],
                "parameters": json.loads(row["params"]), "state": row["state"], "created_at": row["created"],
                "updated_at": row["updated"], "started_at": row["started"], "finished_at": row["finished"],
                "queue_recovered": bool(row["recovered"]), "result": json.loads(row["result"]) if row["result"] else None,
                "error": json.loads(row["error"]) if row["error"] else None, "timings": json.loads(row["timings"])}

    def submit(self, params, idem):
        import hashlib
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        failure = None
        with self.transaction() as db:
            existing = db.execute("SELECT * FROM jobs WHERE idem=?", (idem,)).fetchone()
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise SidecarError("idempotency_conflict", "key was already used for different parameters", 409)
                return self.job_json(existing), True
            if params.get("retry_of"):
                old = db.execute("SELECT state FROM jobs WHERE id=?", (params["retry_of"],)).fetchone()
                if old is None or old[0] not in ("failed", "interrupted", "cancelled"):
                    raise SidecarError("invalid_retry", "retry_of must name a failed, interrupted, or cancelled job", 409)
            if params["operation"] == "matrix.multiply":
                handles = {params["a"], params["b"]}
                a, b = self._object(db, params["a"]), self._object(db, params["b"])
                am, bm = json.loads(a["metadata"]), json.loads(b["metadata"])
                if am.get("format") != "npy" or bm.get("format") != "npy":
                    raise SidecarError("invalid_object_type", "matrix.multiply requires NPY matrix handles", 422)
                if params["a_shape"] != am["shape"] or params["b_shape"] != bm["shape"]:
                    raise SidecarError("shape_mismatch", "declared shapes must match stored objects", 422)
                if am["shape"][1] != bm["shape"][0] or params["dtype"] != "float32":
                    raise SidecarError("invalid_dimensions", "inner dimensions must match and dtype must be float32", 422)
                output = am["shape"][0] * bm["shape"][1] * 4
            elif params["operation"] == "image.embed" and self.config.enable_embeddings:
                handles = set(params["images"])
                if not 1 <= len(params["images"]) <= self.config.max_images:
                    raise SidecarError("image_batch_limit", "image batch exceeds configured limit", 413)
                for handle in handles:
                    obj = self._object(db, handle)
                    if json.loads(obj["metadata"]).get("kind") != "image":
                        raise SidecarError("invalid_object_type", "image.embed requires validated JPEG/PNG image handles", 422)
                output = len(params["images"]) * 576 * 4
            else:
                raise SidecarError("unsupported_operation", "operation is not enabled on this worker", 422)
            if output > self.config.output_limit:
                raise SidecarError("output_limit", "estimated output exceeds configured limit", 413)
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE state='queued'").fetchone()[0]
            if count >= self.config.queue_limit:
                raise SidecarError("queue_full", "waiting queue is full; retry with the same key later", 429)
            reserve = output + 4096
            if not self._room(db, reserve, handles):
                failure = SidecarError("storage_full", "cannot reserve output storage", 507)
            else:
                now, job_id = time.time(), identifier("j_")
                db.execute("INSERT INTO jobs(id,idem,fingerprint,params,state,created,updated,queued_mono) VALUES (?,?,?,?,?,?,?,?)",
                           (job_id, idem, fingerprint, canonical, "queued", now, now, time.monotonic()))
                db.executemany("INSERT INTO job_inputs VALUES (?,?)", [(job_id, h) for h in handles])
                db.execute("INSERT INTO reservations VALUES (?,?)", (job_id, reserve))
                row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if failure:
            raise failure
        return self.job_json(row), False

    def job(self, job_id=None, *, idem=None):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE " + ("idem=?" if idem else "id=?"), (idem or job_id,)).fetchone()
            if row is None:
                raise SidecarError("job_not_found", "unknown job ID or idempotency key", 404)
            return self.job_json(row)

    def jobs(self, limit=50):
        with self.transaction() as db:
            return [self.job_json(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]

    def claim(self):
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM jobs WHERE state IN ('running','cancel_requested')").fetchone():
                return None
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            timings = {"queue_wait_s": max(0.0, time.monotonic() - row["queued_mono"])}
            db.execute("UPDATE jobs SET state='running',started=?,updated=?,timings=? WHERE id=?",
                       (now, now, json.dumps(timings), row["id"]))
            return self.job_json(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def cancel(self, job_id):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise SidecarError("job_not_found", "unknown job", 404)
            now = time.time()
            if row["state"] == "queued":
                db.execute("UPDATE jobs SET state='cancelled',updated=?,finished=? WHERE id=?", (now, now, job_id))
                db.execute("DELETE FROM reservations WHERE id=?", (job_id,))
            elif row["state"] == "running":
                db.execute("UPDATE jobs SET state='cancel_requested',updated=? WHERE id=?", (now, job_id))
            return self.job_json(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def finish(self, job_id, state, timings, *, temporary=None, metadata=None, error=None, worker_started=None):
        commit_started = time.perf_counter()
        digest = None
        if state == "succeeded" and temporary:
            digest = file_hash(temporary)
            timings["result_hash_s"] = time.perf_counter() - commit_started
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["state"] not in ("running", "cancel_requested"):
                return
            result = None
            if row["state"] == "cancel_requested":
                state, error = "cancelled", None
            if state == "succeeded":
                retained = json.loads(row["params"])["retain_output"]
                seconds = self.config.result_retention_seconds if retained else self.config.transient_result_seconds
                result = self._commit_object(db, temporary, digest, metadata, seconds, job_id)
                # Execution provenance belongs to the job even if its bytes deduplicate
                # against an input produced by another device or operation.
                result["metadata"] = metadata
            elif temporary:
                temporary.unlink(missing_ok=True)
            db.execute("DELETE FROM reservations WHERE id=?", (job_id,))
            merged = {**json.loads(row["timings"]), **timings}
            merged["result_commit_s"] = time.perf_counter() - commit_started
            if worker_started is not None:
                merged["worker_total_s"] = time.perf_counter() - worker_started
            now = time.time()
            db.execute("UPDATE jobs SET state=?,updated=?,finished=?,result=?,error=?,timings=? WHERE id=?",
                       (state, now, now, json.dumps(result) if result else None, json.dumps(error) if error else None,
                        json.dumps(merged), job_id))

    def reconcile(self, *, startup=False):
        """Called only after the old executor has exited, while the worker lock is held."""
        with self.transaction() as db:
            now = time.time()
            err = json.dumps({"code": "executor_lost", "message": "executor disappeared; operation did not resume"})
            db.execute("UPDATE jobs SET state='interrupted',finished=?,updated=?,error=? WHERE state IN ('running','cancel_requested')",
                       (now, now, err))
            if startup:
                db.execute("DELETE FROM reservations WHERE id NOT IN (SELECT id FROM jobs WHERE state='queued')")
                db.execute("DELETE FROM leases")
            else:
                # API uploads continue while the executor is replaced.
                db.execute("DELETE FROM reservations WHERE id IN (SELECT id FROM jobs WHERE state != 'queued')")
            queued = db.execute("SELECT * FROM jobs WHERE state='queued'").fetchall()
            for row in queued:
                try:
                    params = json.loads(row["params"])
                    handles = params["images"] if params["operation"] == "image.embed" else [params["a"], params["b"]]
                    for handle in set(handles):
                        obj = self._object(db, handle)
                        path = self.path(obj["sha256"])
                        inspect_object(path, self.config, output=True)
                        if file_hash(path) != obj["sha256"]:
                            raise SidecarError("object_corrupt", "queued input hash changed", 422)
                except SidecarError as exc:
                    db.execute("UPDATE jobs SET state='failed',error=?,finished=?,updated=? WHERE id=?",
                               (json.dumps({"code": exc.code, "message": exc.message}), now, now, row["id"]))
                    db.execute("DELETE FROM reservations WHERE id=?", (row["id"],))
                else:
                    db.execute("UPDATE jobs SET queued_mono=?,recovered=1 WHERE id=?", (time.monotonic(), row["id"]))
        for path in self.temp.iterdir():
            if path.is_file():
                if startup or path.name.startswith("j_"):
                    path.unlink(missing_ok=True)
        if startup:
            with self.transaction() as db:
                known = {r[0] for r in db.execute("SELECT sha256 FROM objects WHERE available=1")}
            for path in self.blobs.glob("*.npy"):
                if path.stem not in known:
                    path.unlink(missing_ok=True)

    def set_runtime(self, payload):
        with self.transaction() as db:
            db.execute("INSERT OR REPLACE INTO runtime VALUES (1,?)", (json.dumps(payload),))

    def runtime(self):
        with self.transaction() as db:
            row = db.execute("SELECT payload FROM runtime WHERE id=1").fetchone()
            return json.loads(row[0]) if row else {}

    def metrics(self):
        with self.transaction() as db:
            states = dict(db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state").fetchall())
            count, size = db.execute("SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM objects WHERE available=1").fetchone()
            reserved = db.execute("SELECT COALESCE(SUM(bytes),0) FROM reservations").fetchone()[0]
            return {"jobs": states, "queue_depth": states.get("queued", 0), "object_count": count,
                    "storage_bytes": size, "reserved_bytes": reserved, "disk_budget_bytes": self.config.disk_budget,
                    "counters": dict(db.execute("SELECT name,value FROM counters").fetchall())}
