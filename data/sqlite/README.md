# SQLite maintenance note

`gempa.db` on the current WSL `/mnt/e` mount can fail WAL checkpoint with `disk I/O error` even though immutable reads work. A rebuilt copy may be produced as `gempa.rebuilt.tmp.db`/`gempa.db.new`, but replacing the open/locked Windows-mounted file can require closing external handles or moving the project to the Linux filesystem.

Safe recovery steps:

```bash
python scripts/maintain_sqlite.py --checkpoint --integrity
# if checkpoint fails:
python - <<'PY'
import sqlite3, pathlib
src=pathlib.Path('data/sqlite/gempa.db')
out=pathlib.Path('data/sqlite/gempa.rebuilt.db')
uri=f'file:{src.resolve()}?mode=ro&immutable=1'
con=sqlite3.connect(uri, uri=True)
dst=sqlite3.connect(out)
con.backup(dst)
dst.close(); con.close()
print(sqlite3.connect(out).execute('pragma integrity_check').fetchone()[0])
PY
# stop all app/Python processes, then replace gempa.db with gempa.rebuilt.db and remove gempa.db-wal/gempa.db-shm
```
