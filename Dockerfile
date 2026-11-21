FROM rust:1-bullseye as builder

WORKDIR /app
RUN apt-get update && apt-get install -y libssl-dev pkg-config

COPY Cargo.toml Cargo.lock* ./
COPY crates ./crates

RUN cargo build --release -p haven-cli

FROM debian:bullseye-slim
WORKDIR /app
RUN apt-get update && apt-get install -y ca-certificates libssl1.1 sqlite3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/haven-server /usr/local/bin/haven-server
COPY public ./public

ENV PORT=3000
ENV HOST=0.0.0.0
ENV HAVEN_DB_PATH=/data/haven.db
ENV HAVEN_PUBLIC_DIR=/app/public

EXPOSE 3000
VOLUME ["/data"]

CMD ["haven-server"]
