//! # da-harness
//!
//! A framework for running LLM-driven agent loops with structured JSON
//! request/response schemas.

pub mod multi_tool;
pub mod single_tool;

// Re-export types needed by callers constructing user messages and tool schemas.
pub use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestDeveloperMessage, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestFunctionMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionRequestToolMessage, ChatCompletionRequestToolMessageContent,
    ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionToolType, FunctionCall,
};
pub use schemars;

use anyhow::Context;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionResponseMessage, ChatCompletionTool, CompletionUsage,
        CreateChatCompletionRequestArgs,
    },
};
use tracing::warn;

// ─── Token usage ───────────────────────────────────────────────────────────

/// Token counts reported by an OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenUsage {
    /// Tokens in the prompt (input / context).
    pub prompt_tokens: u32,
    /// Tokens in the generated completion (output).
    pub completion_tokens: u32,
    /// Total tokens for the request (`prompt + completion` when both reported).
    pub total_tokens: u32,
    /// Cached prompt tokens when the server reports `prompt_tokens_details`.
    pub cached_prompt_tokens: Option<u32>,
}

impl TokenUsage {
    /// Build from `async_openai`'s [`CompletionUsage`], if present.
    pub fn from_completion_usage(usage: &CompletionUsage) -> Self {
        Self {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cached_prompt_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens),
        }
    }
}

// ─── OpenAIClient ──────────────────────────────────────────────────────

/// Configuration for connecting to an OpenAI-compatible LLM endpoint.
#[derive(Debug, Clone)]
pub struct LLMConfig {
    /// The base URL of the API endpoint (e.g., `http://127.0.0.1:4242/v1`).
    pub api_base: String,
    /// The model identifier to use for completions.
    pub model_name: String,
    /// The API key for authentication. May be empty for local endpoints.
    pub api_key: String,
    /// Optional explicit maximum context window size in tokens for this model.
    ///
    /// When set, this value is used for compaction threshold calculations
    /// instead of discovering it from the server's `/models` endpoint.
    /// Useful for testing compaction with a deliberately small window.
    pub max_context_tokens: Option<usize>,
}

impl Default for LLMConfig {
    /// Returns a default configuration pointing to a local endpoint at
    /// `http://127.0.0.1:4242/v1` with the model name `LocalModel`.
    fn default() -> Self {
        Self {
            api_base: "http://127.0.0.1:4242/v1".to_owned(),
            model_name: "LocalModel".to_owned(),
            api_key: "".to_owned(),
            max_context_tokens: None,
        }
    }
}

/// A client for OpenAI-compatible chat completion APIs.
///
/// Wraps `async_openai::Client` with convenience methods for health checking
/// and readiness polling.
///
/// The client can optionally know the model's maximum context window size
/// (in tokens), either from [`LLMConfig::max_context_tokens`] or discovered
/// at runtime by querying the `/models` endpoint. This information is used
/// by [`run_loop_with_max_and_context`] (and friends) to trigger automatic
/// history compaction when context usage approaches the limit.
#[derive(Clone)]
pub struct OpenAIClient {
    client: Client<OpenAIConfig>,
    model_name: String,
    max_context_tokens: Option<usize>,
}

impl Default for OpenAIClient {
    /// Creates a client with the default [`LLMConfig`].
    fn default() -> Self {
        Self::with_config(LLMConfig::default())
    }
}

impl OpenAIClient {
    /// Creates a new client with the default [`LLMConfig`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new client with the given configuration.
    pub fn with_config(config: LLMConfig) -> Self {
        let mut async_config = OpenAIConfig::default().with_api_key(config.api_key.clone());

        if !config.api_base.is_empty() {
            let base = config.api_base.replace("/chat/completions", "");
            async_config = async_config.with_api_base(base);
        }

        Self {
            client: Client::with_config(async_config),
            model_name: config.model_name,
            max_context_tokens: config.max_context_tokens,
        }
    }

    /// Returns the model name configured for this client.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Returns the known maximum context window size in tokens, if known.
    ///
    /// This may have been provided at construction time via [`LLMConfig`]
    /// or populated by a prior call to [`OpenAIClient::discover_max_context_tokens`].
    pub fn max_context_tokens(&self) -> Option<usize> {
        self.max_context_tokens
    }

    /// Queries the server's `/models` endpoint for the configured model and
    /// attempts to extract its maximum context window length.
    ///
    /// Many OpenAI-compatible servers (notably vLLM) include an extension field
    /// such as `max_model_len` (or `context_length`, `n_ctx`, etc.) on the model
    /// object returned by `/models`. This method parses the raw response (the
    /// typed `async_openai` model object only contains the standard four fields)
    /// and looks for several common field names.
    ///
    /// If a positive integer value is found for the current `model_name`, it is
    /// cached in the client and returned as `Some(n)`.
    ///
    /// Returns `Ok(None)` if the model was listed but no recognized context field
    /// was present. Returns an error only on transport / parse failures.
    ///
    /// This is called automatically by the `run_loop*` entry points when no
    /// explicit max context is supplied.
    pub async fn discover_max_context_tokens(&mut self) -> anyhow::Result<Option<usize>> {
        use async_openai::config::Config;

        let cfg = self.client.config();
        let url = cfg.url("/models");
        let headers = cfg.headers();
        let qparams = cfg.query();

        let resp = reqwest::Client::new()
            .get(url)
            .headers(headers)
            .query(&qparams)
            .send()
            .await
            .context("sending GET /models for context discovery")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("GET /models failed: {} body: {}", status, body);
        }

        let json: serde_json::Value = resp
            .json()
            .await
            .context("decoding /models response as JSON")?;

        let data = match json.get("data").and_then(|d| d.as_array()) {
            Some(arr) => arr,
            None => return Ok(None),
        };

        let target = self.model_name.as_str();
        for entry in data {
            if entry.get("id").and_then(|i| i.as_str()) != Some(target) {
                continue;
            }
            // Look for common extension fields used by vLLM, SGLang, and others.
            for key in [
                "max_model_len",
                "max_context_length",
                "context_length",
                "n_ctx",
                "max_seq_len",
                "context_window",
                "max_tokens",
            ] {
                if let Some(n) = entry.get(key).and_then(|v| v.as_u64())
                    && n > 0
                {
                    let n = n as usize;
                    self.max_context_tokens = Some(n);
                    return Ok(Some(n));
                }
            }
            // Model found but no recognized context field.
            return Ok(None);
        }
        Ok(None)
    }

    /// Sends a chat completion request and returns the assistant's text response.
    ///
    /// This is a convenience wrapper around [`OpenAIClient::chat_with_usage`]
    /// that discards token usage information.
    pub async fn chat(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        temperature: f32,
    ) -> anyhow::Result<String> {
        self.chat_with_usage(messages, temperature)
            .await
            .map(|(text, _)| text)
    }

    /// Sends a chat completion request and returns the assistant's text response
    /// along with token usage (if the server reported it).
    ///
    /// The returned [`TokenUsage`] (when present) is the primary signal used by
    /// the single-tool run loop to decide when to trigger compaction
    /// (`prompt_tokens` vs context window).
    pub async fn chat_with_usage(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        temperature: f32,
    ) -> anyhow::Result<(String, Option<TokenUsage>)> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model_name())
            .messages(messages)
            .temperature(temperature)
            .build()?;

        let response = self.client.chat().create(request).await?;

        let text = response.choices[0]
            .message
            .content
            .clone()
            .ok_or_else(|| anyhow::anyhow!("LLM returned no content"))?;

        let usage = response
            .usage
            .as_ref()
            .map(TokenUsage::from_completion_usage);
        Ok((text, usage))
    }

    /// Sends a chat completion request with tools and returns the full response
    /// message (including `tool_calls`) along with token usage when reported.
    pub async fn chat_with_tools(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Vec<ChatCompletionTool>,
        parallel_tool_calls: bool,
        temperature: f32,
    ) -> anyhow::Result<(ChatCompletionResponseMessage, Option<TokenUsage>)> {
        let mut request = CreateChatCompletionRequestArgs::default()
            .model(self.model_name())
            .messages(messages)
            .temperature(temperature)
            .tools(tools)
            .build()?;

        request.parallel_tool_calls = Some(parallel_tool_calls);

        let response = self.client.chat().create(request).await?;

        let usage = response
            .usage
            .as_ref()
            .map(TokenUsage::from_completion_usage);
        Ok((response.choices[0].message.clone(), usage))
    }

    /// Checks whether the LLM endpoint is reachable by sending a minimal
    /// chat completion request.
    ///
    /// # Errors
    /// Returns an error if the API call fails.
    pub async fn check_health(&self) -> anyhow::Result<()> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model_name())
            .messages([
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a test assistant.")
                    .build()?
                    .into(),
                ChatCompletionRequestUserMessageArgs::default()
                    .content("Say 'OK'.")
                    .build()?
                    .into(),
            ])
            .temperature(0.0)
            .build()?;

        let _ = self.client.chat().create(request).await?;
        Ok(())
    }

    /// Polls the LLM endpoint at 30-second intervals until it becomes
    /// available, or indefinitely if it never responds.
    ///
    /// # Errors
    /// This function does not return an error; it loops until `check_health`
    /// succeeds. Warnings are logged on each failed attempt via `tracing`.
    pub async fn wait_until_ready(&self) -> anyhow::Result<()> {
        use std::time::Duration;
        use tokio::time::sleep;

        let poll_interval = Duration::from_secs(30);
        let mut attempt = 0;

        loop {
            attempt += 1;
            match self.check_health().await {
                Ok(()) => return Ok(()),
                Err(e) => {
                    warn!("LLM check attempt {} failed: {}", attempt, e);
                    sleep(poll_interval).await;
                }
            }
        }
    }
}

use async_openai::types::FunctionObjectArgs;
use schemars::r#gen::SchemaGenerator;

/// Extracts the JSON content from an LLM reply that may be wrapped in
/// markdown code fences (e.g. "```json\n{...}\n```" or "```\n{...}\n```").
///
/// Many local models emit fenced JSON even when instructed to output raw JSON.
/// This helper makes the harness tolerant of that common behavior while
/// preserving the original text for error reporting.
pub fn extract_json(text: &str) -> &str {
    let t = text.trim();
    if let Some(start) = t.find("```") {
        let mut rest = &t[start + 3..];
        // Skip optional language specifier: json, JSON, Json, etc.
        rest = rest.trim_start();
        if rest.len() >= 4 {
            let prefix = &rest[..4].to_ascii_lowercase();
            if prefix == "json" || prefix.starts_with("json") {
                rest = rest[4..].trim_start();
            }
        }
        if let Some(end) = rest.find("```") {
            return rest[..end].trim();
        }
        // No closing fence; return everything after the opening fence.
        return rest.trim();
    }
    t
}

pub fn generate_tool_schema<T>() -> anyhow::Result<ChatCompletionTool>
where
    T: schemars::JsonSchema,
{
    let mut schema_gen = SchemaGenerator::default();
    let schema = <T as schemars::JsonSchema>::json_schema(&mut schema_gen);
    let definitions = schema_gen.take_definitions();

    let mut schema_obj = serde_json::to_value(&schema)?
        .as_object()
        .cloned()
        .unwrap_or_default();

    if !definitions.is_empty() {
        schema_obj.insert("definitions".into(), serde_json::to_value(&definitions)?);
    }

    // Remove metadata that's already represented at the function level
    schema_obj.remove("description");
    schema_obj.remove("title");

    // Add $schema from schemars settings
    if let Some(schema_url) = schema_gen.settings().meta_schema.clone() {
        schema_obj.insert("$schema".into(), serde_json::Value::String(schema_url));
    }

    let type_name = std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap_or("Unknown")
        .to_string();

    let description = match &schema {
        schemars::schema::Schema::Object(obj) => {
            obj.metadata.as_ref().and_then(|m| m.description.clone())
        }
        _ => None,
    };

    let mut func_args = FunctionObjectArgs::default();
    func_args.name(type_name);
    if let Some(desc) = description {
        func_args.description(desc);
    }
    func_args.parameters(serde_json::Value::Object(schema_obj));

    let tool = ChatCompletionTool {
        r#type: async_openai::types::ChatCompletionToolType::Function,
        function: func_args.build().context("building tool schema")?,
    };

    Ok(tool)
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
    enum SharedAction {
        Query,
        Stop,
    }

    #[derive(Debug, Clone, Serialize, schemars::JsonSchema)]
    struct TestRequest {
        action: SharedAction,
        data: String,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
    struct TestResponse {
        action: SharedAction,
        result: String,
    }

    #[test]
    fn test_tool_schema() {
        /// Looks up the current weather for a given location.
        #[allow(dead_code)]
        #[derive(JsonSchema, Clone)]
        pub struct GetWeather {
            /// The city and country, e.g. "London, UK" or "Tokyo, Japan".
            pub location: String,
            /// The temperature unit to use: "celsius" or "fahrenheit".
            pub unit: String,
        }

        println!(
            "{}",
            serde_json::to_string_pretty(&generate_tool_schema::<GetWeather>().unwrap()).unwrap()
        );
    }

    #[test]
    fn test_shared_type_deduplicated() {
        let mut schema_gen = SchemaGenerator::default();
        let request_schema = <TestRequest as schemars::JsonSchema>::json_schema(&mut schema_gen);
        let response_schema = <TestResponse as schemars::JsonSchema>::json_schema(&mut schema_gen);

        let definitions = schema_gen.take_definitions();

        assert!(
            definitions.contains_key("SharedAction"),
            "SharedAction should be in definitions"
        );

        let mut schema_obj = serde_json::Map::new();
        if !definitions.is_empty() {
            schema_obj.insert(
                "definitions".into(),
                serde_json::to_value(&definitions).unwrap(),
            );
        }
        schema_obj.insert(
            "Request".into(),
            serde_json::to_value(&request_schema).unwrap(),
        );
        schema_obj.insert(
            "Response".into(),
            serde_json::to_value(&response_schema).unwrap(),
        );

        let combined = serde_json::Value::Object(schema_obj.clone());
        let json = serde_json::to_string_pretty(&combined).unwrap();
        eprintln!("{}", json);

        let req_str = serde_json::to_string(&schema_obj["Request"]).unwrap();
        let resp_str = serde_json::to_string(&schema_obj["Response"]).unwrap();

        assert!(
            req_str.contains("SharedAction"),
            "Request should reference SharedAction: {}",
            req_str
        );
        assert!(
            resp_str.contains("SharedAction"),
            "Response should reference SharedAction: {}",
            resp_str
        );
    }
}
