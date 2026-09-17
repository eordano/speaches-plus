pub mod stages;
pub mod workspace;

pub use stages::{Clip, Endpoints};
pub use workspace::{Manifest, Recording, Registry, Speaker, Workspace};

#[cfg(test)]
mod tests;
