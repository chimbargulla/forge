use std::sync::Arc;

use anyhow::Context;
use forge_app::ProviderRegistry;
use forge_app::domain::{Provider, ProviderUrl};
use forge_app::dto::AppConfig;
use tokio::sync::RwLock;

use crate::EnvironmentInfra;

type ProviderSearch = (&'static str, Box<dyn FnOnce(&str) -> Provider>);

pub struct ForgeProviderRegistry<F> {
    infra: Arc<F>,
    // IMPORTANT: This cache is used to avoid logging out if the user has logged out from other
    // session. This helps to keep the user logged in for current session.
    cache: Arc<RwLock<Option<Provider>>>,
}

impl<F: EnvironmentInfra> ForgeProviderRegistry<F> {
    pub fn new(infra: Arc<F>) -> Self {
        Self { infra, cache: Arc::new(Default::default()) }
    }

    fn provider_url(&self) -> Option<ProviderUrl> {
        if let Some(url) = self.infra.get_env_var("OPENAI_URL") {
            return Some(ProviderUrl::OpenAI(url));
        }

        // Check for Anthropic URL override
        if let Some(url) = self.infra.get_env_var("ANTHROPIC_URL") {
            return Some(ProviderUrl::Anthropic(url));
        }
        None
    }
    fn get_provider(&self, _forge_config: AppConfig) -> Option<Provider> {
        // if let Some(forge_key) = &forge_config.key_info {
        //     let provider = Provider::forge(forge_key.api_key.as_str());
        //     return Some(override_url(provider, self.provider_url()));
        // }
        resolve_env_provider(self.provider_url(), self.infra.as_ref())
    }
}

#[async_trait::async_trait]
impl<F: EnvironmentInfra> ProviderRegistry for ForgeProviderRegistry<F> {
    async fn get_provider(&self, config: AppConfig) -> anyhow::Result<Provider> {
        if let Some(provider) = self.cache.read().await.as_ref() {
            return Ok(provider.clone());
        }

        let provider = self
            .get_provider(config)
            .context("No valid provider configuration found. Please set one of the following environment variables: OPENROUTER_API_KEY, REQUESTY_API_KEY, XAI_API_KEY, CHUTES_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY. For more details, visit: https://forgecode.dev/docs/custom-providers/")?;
        self.cache.write().await.replace(provider.clone());
        Ok(provider)
    }
}

fn resolve_env_provider<F: EnvironmentInfra>(
    url: Option<ProviderUrl>,
    env: &F,
) -> Option<Provider> {
    let keys: [ProviderSearch; 6] = [
        // ("FORGE_KEY", Box::new(Provider::forge)),
        ("OPENROUTER_API_KEY", Box::new(Provider::open_router)),
        ("REQUESTY_API_KEY", Box::new(Provider::requesty)),
        ("XAI_API_KEY", Box::new(Provider::xai)),
        ("CHUTES_API_KEY", Box::new(Provider::chutes)),
        ("OPENAI_API_KEY", Box::new(Provider::openai)),
        ("ANTHROPIC_API_KEY", Box::new(Provider::anthropic)),
    ];

    keys.into_iter().find_map(|(key, fun)| {
        env.get_env_var(key).map(|key| {
            let provider = fun(&key);
            override_url(provider, url.clone())
        })
    })
}

fn override_url(mut provider: Provider, url: Option<ProviderUrl>) -> Provider {
    if let Some(url) = url {
        provider.url(url);
    }
    provider
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use forge_app::domain::{Environment, HttpConfig, RetryConfig};
    use super::*;

    struct MockEnv {
        vars: HashMap<String, String>,
    }

    impl MockEnv {
        fn new() -> Self {
            Self {
                vars: HashMap::new(),
            }
        }

        fn set_var(&mut self, key: &str, value: &str) {
            self.vars.insert(key.to_string(), value.to_string());
        }
    }

    impl EnvironmentInfra for MockEnv {
        fn get_environment(&self) -> Environment {
            use std::path::PathBuf;
            use url::Url;

            Environment {
                os: "test".to_string(),
                pid: 1,
                cwd: PathBuf::from("/test"),
                home: None,
                shell: "bash".to_string(),
                base_path: PathBuf::from("/test"),
                forge_api_url: Url::parse("https://test.com").unwrap(),
                retry_config: RetryConfig::default(),
                max_search_lines: 1000,
                max_search_result_bytes: 1000000,
                fetch_truncation_limit: 100000,
                stdout_max_prefix_length: 500,
                stdout_max_suffix_length: 500,
                stdout_max_line_length: 1000,
                max_read_size: 1000000,
                http: HttpConfig::default(),
                max_file_size: 10000000,
                tool_timeout: 30,
            }
        }

        fn get_env_var(&self, key: &str) -> Option<String> {
            self.vars.get(key).cloned()
        }
    }

    #[test]
    fn test_chutes_api_key_resolution() {
        let mut env = MockEnv::new();
        env.set_var("CHUTES_API_KEY", "test_chutes_key");

        let provider = resolve_env_provider(None, &env).unwrap();
        
        assert!(provider.is_chutes());
        assert_eq!(provider.key(), Some("test_chutes_key"));
    }

    #[test]
    fn test_provider_priority_order() {
        let mut env = MockEnv::new();
        // Set multiple keys to test priority
        env.set_var("OPENROUTER_API_KEY", "openrouter_key");
        env.set_var("CHUTES_API_KEY", "chutes_key");
        env.set_var("OPENAI_API_KEY", "openai_key");

        let provider = resolve_env_provider(None, &env).unwrap();
        
        // Should prioritize OpenRouter (first in the list)
        assert!(provider.is_open_router());
        assert_eq!(provider.key(), Some("openrouter_key"));
    }
}
