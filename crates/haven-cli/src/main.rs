use clap::Parser;
use dashmap::DashMap;
use haven_core::{db::create_pool, models::PublicConfig};
use haven_server::app::{create_router, AppState};
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

#[derive(Parser, Debug)]
#[command(
    name = "haven-server",
    author = "Khanhs <khanhtg2011@gmail.com>",
    version = "4.29.0",
    about = "Haven Server - Self-hosted private chat backend, your server, your rules"
)]
pub struct Cli {
    #[arg(short, long, default_value = "3000", env = "PORT")]
    pub port: u16,

    #[arg(short = 'H', long, default_value = "0.0.0.0", env = "HOST")]
    pub host: String,

    #[arg(short, long, default_value = "./data/haven.db", env = "HAVEN_DB_PATH")]
    pub db: String,

    #[arg(
        long,
        default_value = "haven_default_secret_change_me_in_prod",
        env = "JWT_SECRET"
    )]
    pub jwt_secret: String,

    #[arg(long, default_value = "./public", env = "HAVEN_PUBLIC_DIR")]
    pub public_dir: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if let Some(parent) = std::path::Path::new(&cli.db).parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    tracing::info!(
        host = %cli.host,
        port = cli.port,
        db = %cli.db,
        public_dir = %cli.public_dir,
        "Initializing Haven SQLite database & static assets"
    );
    let pool = create_pool(&cli.db)?;

    let (tx_broadcast, _) = broadcast::channel(1024);
    let state = Arc::new(AppState {
        pool,
        jwt_secret: cli.jwt_secret.into_bytes(),
        public_config: Arc::new(RwLock::new(PublicConfig::default())),
        voice_rooms: DashMap::new(),
        tx_broadcast,
        public_dir: cli.public_dir,
    });

    let app = create_router(state);
    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("⬡ Haven server listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
