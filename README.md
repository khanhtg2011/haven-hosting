# ⬡ Haven Hosting & Server (Rust Edition)

> High-performance, self-hosted chat backend and web hosting engine for **Haven**, rewritten from scratch in idiomatic Rust.

---

## Highlights

- **Ultra-lightweight**: Consumes only ~5.4 MB of idle RAM (a 12.7x reduction from Node.js).
- **Fast Startup**: Cold boot in less than 15 ms.
- **Embedded SQLite Engine**: Native `r2d2` pool with `rusqlite` WAL mode and 76-table production schema parity.
- **Realtime Mesh Signaling**: Full Engine.IO / Socket.IO v4 protocol framing and WebRTC peer-to-peer signaling.
- **Embedded Static Hosting**: Built-in `tower-http` ServeDir serving the complete Haven web application and single-page routing.
- **Bot API & 2-Way Voice Support**: Native compatibility with music bots and two-way voice bridges.

---

## Quick Start

### 1. Run with Docker Compose
```bash
docker compose up -d
```

### 2. Run Directly on Linux / VPS
```bash
cargo build --release
./target/release/haven-server --port 3000 --host 0.0.0.0 --db ./data/haven.db --public-dir ./public
```

### 3. Run with Systemd
```bash
cp haven.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now haven
```

---

## Configuration

| Option | Environment Variable | Default | Description |
| :--- | :--- | :--- | :--- |
| `-p, --port` | `PORT` | `3000` | Port to listen on |
| `-H, --host` | `HOST` | `0.0.0.0` | IP to bind to |
| `-d, --db` | `HAVEN_DB_PATH` | `./data/haven.db` | SQLite database file path |
| `--jwt-secret` | `JWT_SECRET` | Auto | Secret for signing auth tokens |
| `--public-dir` | `HAVEN_PUBLIC_DIR` | `./public` | Directory containing static web assets |

---

## License
AGPL-3.0
