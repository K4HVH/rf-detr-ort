use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Model file not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model: {0}")]
    InvalidModel(String),

    #[error("Batch too large: requested {requested}, max {max}")]
    BatchTooLarge { requested: usize, max: usize },

    #[error("All execution providers failed to initialize")]
    NoProviderAvailable,

    #[error("Session build error: {0}")]
    SessionBuild(String),
}

pub type Result<T> = std::result::Result<T, Error>;
