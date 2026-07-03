//! # da-harness
//!
//! A framework for running LLM-driven agent loops with structured JSON
//! request/response schemas.
//!
//! ## Overview
//!
//! `da-harness` provides a simple abstraction over OpenAI-compatible chat APIs
//! to build agents that operate in iterative loops. Each iteration the agent
//! sends a structured JSON response, the controller executes it, and feeds the
//! result back as the next request. The loop continues until the agent signals
//! completion.
//!
//! ## Core Concepts
//!
//! - **`AgentLoop`** — A trait that defines an agent's behavior: system/user
//!   prompts, example pairs, initial input, and per-iteration control logic.
//! - **`LoopControl`** — An enum returned by each iteration to either continue
//!   the loop with a new request or stop with a final output value.
//! - **`run_loop`** — The main execution function that drives the agent loop,
//!   building prompts from JSON schemas (via `schemars`), sending them to the
//!   LLM, deserializing responses, and managing conversation history.
//! - **`OpenAIClient`** — A wrapper around `async-openai` for chat completions,
//!   with health checking and readiness polling.
//!
//! ## Example
//!
//! ```ignore
//! use da_harness::{AgentLoop, LoopConfigBuilder, LoopControl, OpenAIClient, run_loop};
//! use async_trait::async_trait;
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Debug, Clone, Serialize, JsonSchema)]
//! struct MyRequest {
//!     step: u32,
//!     question: String,
//! }
//!
//! #[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
//! struct MyResponse {
//!     answer: String,
//!     done: bool,
//! }
//!
//! struct MyAgent;
//!
//! #[async_trait]
//! impl AgentLoop for MyAgent {
//!     type Request = MyRequest;
//!     type Response = MyResponse;
//!     type Output = String;
//!
//!     fn system_prompt(&self) -> &str {
//!         "You are a helpful assistant."
//!     }
//!
//!     fn user_prompt(&self) -> &str {
//!         "Answer the following question. Set done to true when finished."
//!     }
//!
//!     fn examples(&self) -> Vec<(Self::Request, Self::Response)> {
//!         vec![]
//!     }
//!
//!     fn initial_input(&self) -> Self::Request {
//!         MyRequest { step: 1, question: "What is 6×7?".into() }
//!     }
//!
//!     async fn iteration(
//!         &self,
//!         response: Self::Response,
//!         _history: &[(Self::Request, Self::Response)],
//!     ) -> LoopControl<Self> {
//!         if response.done {
//!             LoopControl::Stop(response.answer)
//!         } else {
//!             LoopControl::Continue(MyRequest {
//!                 step: 2,
//!                 question: "Now square that result.".into(),
//!             })
//!         }
//!     }
//! }
//!
//! # #[tokio::main]
//! # async fn main() {
//! let client = OpenAIClient::new();
//! let config = LoopConfigBuilder::default().try_build().unwrap();
//! let output = run_loop(client, MyAgent, config).await.unwrap();
//! println!("Result: {}", output);
//! # }
//! ```

use anyhow::Context;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionTool, CreateChatCompletionRequestArgs,
        FunctionObjectArgs,
    },
};
use async_trait::async_trait;
use schemars::r#gen::SchemaGenerator;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

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
    /// along with the number of prompt tokens used for the request (if the server
    /// reported usage).
    ///
    /// The returned `Option<u32>` is the `prompt_tokens` value from the
    /// `CompletionUsage` object in the response (when present). This is the
    /// primary signal used by the run loop to decide when to trigger compaction.
    pub async fn chat_with_usage(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        temperature: f32,
    ) -> anyhow::Result<(String, Option<u32>)> {
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

        let prompt_tokens = response.usage.map(|u| u.prompt_tokens);
        Ok((text, prompt_tokens))
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

// ─── LoopControl ──────────────────────────────────────────────────────

/// Controls the flow of an [`AgentLoop`] iteration.
///
/// Returned by [`AgentLoop::iteration`] to signal whether the loop should
/// continue with a new request or stop with a final output value.
pub enum LoopControl<Agent: AgentLoop> {
    /// Continue the loop with the given request for the next iteration.
    Continue(Agent::Request),
    /// Stop the loop and return the given output value.
    Stop(Agent::Output),
}

// ─── CompactPolicy ────────────────────────────────────────────────────

/// Controls how the maximum context window size is determined for compaction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompactPolicy {
    /// Do not compact at all; the loop will never trigger summarization.
    NoCompact,
    /// Discover the model's context window from the server (via `/models`) or
    /// from [`LLMConfig::max_context_tokens`]. This is the default.
    #[default]
    ModelDefault,
    /// Use an explicit token budget, bypassing both config and discovery.
    /// Useful for testing compaction with a deliberately small window.
    Override(usize),
}

// ─── SavePoint ────────────────────────────────────────────────────────

/// Indicates the type of checkpoint being saved by [`AgentLoop::save`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SavePoint {
    /// Normal mid-loop save: the agent is continuing to the next iteration.
    Continue,
    /// The most recent iteration triggered compaction; history has been
    /// truncated and a summary is now in effect.
    Compacted,
    /// Final save after the agent returned [`LoopControl::Stop`]. No further
    /// iterations will occur.
    Final,
}

/// State restored by [`AgentLoop::restore`] to resume an agent loop from a
/// previously saved checkpoint.
#[derive(Debug, Clone)]
pub struct RestoreState<Request, Response> {
    /// Conversation history at the time of the last save.
    pub history: Vec<(Request, Response)>,
    /// The request that was pending when the checkpoint was taken.
    pub current_request: Request,
    /// The "previous sessions summary" if compaction had occurred before the
    /// checkpoint was saved.
    pub previous_summary: Option<String>,
}

// ─── LoopConfig ───────────────────────────────────────────────────────

/// Configuration for a single invocation of [`run_loop`].
///
/// Use the generated [`LoopConfigBuilder`] to set only the fields you care
/// about; missing values fall back to sensible defaults.
///
/// # Example
///
/// ```ignore
/// // Default: no iteration cap, auto-discover context window.
/// let config = LoopConfigBuilder::default().try_build().unwrap();
///
/// // With iteration cap and explicit small context for testing compaction:
/// let config = LoopConfigBuilder::default()
///     .max_iterations(Some(30))
///     .compact_policy(CompactPolicy::Override(700))
///     .try_build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Default, derive_builder::Builder)]
#[builder(build_fn(name = "try_build"))]
pub struct LoopConfig {
    /// Maximum number of agent iterations before the loop fails with an error.
    ///
    /// When `None`, the loop runs indefinitely until the agent returns
    /// [`LoopControl::Stop`].
    ///
    /// Default: `None` (no iteration cap).
    #[builder(default)]
    pub max_iterations: Option<usize>,

    /// Controls how the maximum context window size is determined for compaction.
    ///
    /// Default: [`CompactPolicy::ModelDefault`] (discover from server or config).
    #[builder(default)]
    pub compact_policy: CompactPolicy,
}

// ─── AgentLoop Trait ──────────────────────────────────────────────────

/// Defines the contract for an LLM-driven agent that operates in an iterative loop.
///
/// Implementors specify typed request and response structures (both must be
/// serializable and implement [`schemars::JsonSchema`]), along with prompts,
/// examples, and per-iteration control logic. The [`run_loop`] function drives
/// the loop by building prompts from JSON schemas, sending them to the LLM,
/// and routing responses through [`AgentLoop::iteration`].
///
/// # Associated Types
///
/// - **`Request`** — The type sent *to* the LLM each iteration. Must implement
///   `Serialize`, `Clone`, and `JsonSchema`.
/// - **`Response`** — The type returned *from* the LLM each iteration. Must
///   implement `Deserialize`, `Serialize`, `Clone`, and `JsonSchema`.
/// - **`Output`** — The final result type returned when the loop stops.
#[async_trait]
pub trait AgentLoop: Sized {
    /// The request type sent to the LLM each iteration.
    type Request: Serialize + Clone + schemars::JsonSchema;

    /// The response type returned from the LLM each iteration.
    type Response: for<'de> Deserialize<'de> + Serialize + Clone + schemars::JsonSchema;

    /// The final output type returned when the loop completes.
    type Output;

    /// Returns the system prompt that sets the agent's role and behavior.
    fn system_prompt(&self) -> &str;

    /// Returns the user-facing instruction text, prepended to each request.
    fn user_prompt(&self) -> &str;

    /// Returns example request/response pairs for few-shot prompting.
    /// An empty vector is valid and will omit the examples section.
    fn examples(&self) -> Vec<(Self::Request, Self::Response)>;

    /// Returns the initial request to start the agent loop.
    fn initial_input(&self) -> Self::Request;

    /// Optionally returns additional context to append to the prompt based on
    /// conversation history. Return `None` to omit this section.
    ///
    /// This hook is useful for injecting dynamic instructions, constraints,
    /// or summaries as the conversation progresses.
    fn extend_prompt(&self, _history: &[(Self::Request, Self::Response)]) -> Option<String> {
        None
    }

    /// Processes the LLM's response and decides whether to continue or stop
    /// the loop.
    ///
    /// # Arguments
    /// * `response` — The deserialized response from the LLM.
    /// * `history` — The full conversation history up to and including this iteration.
    ///
    /// # Returns
    /// - [`LoopControl::Continue`] with the next request to keep the loop going.
    /// - [`LoopControl::Stop`] with the final output to end the loop.
    async fn iteration(
        &self,
        response: Self::Response,
        history: &[(Self::Request, Self::Response)],
    ) -> LoopControl<Self>;

    /// Called once before the loop begins to allow the agent to restore
    /// previously saved state.
    ///
    /// Return `Some((history, current_request, summary))` to resume from a
    /// prior checkpoint. The returned history and request replace the initial
    /// empty history and [`AgentLoop::initial_input`]. The optional summary
    /// is injected as the "previous sessions summary" in subsequent prompts.
    ///
    /// Return `None` to start fresh with an empty history and the agent's
    /// initial input. This is the default implementation.
    async fn restore(&mut self) -> Option<RestoreState<Self::Request, Self::Response>> {
        None
    }

    /// Called at the beginning of each iteration (starting from iteration 1)
    /// and once more when the loop is about to stop, allowing the agent to
    /// persist its current state.
    ///
    /// # Arguments
    /// * `history` — The full conversation history up to the current point.
    /// * `summary` — The current "previous sessions summary", if compaction
    ///   has occurred.
    /// * `current_request` — The request that will be (or was just) sent to
    ///   the LLM in this iteration.
    /// * `save_point` — Indicates the type of checkpoint being saved.
    ///
    /// The default implementation does nothing. Implementors are responsible
    /// for all serialization and storage logic.
    async fn save(
        &mut self,
        _history: &[(Self::Request, Self::Response)],
        _summary: Option<&str>,
        _current_request: &Self::Request,
        _save_point: SavePoint,
    ) {
    }
}

// ─── run_loop ────────────────────────────────────────────────────────

/// Runs an [`AgentLoop`] with the given [`OpenAIClient`], agent, and
/// [`LoopConfig`].
///
/// # Compaction
///
/// The [`CompactPolicy`] from the config controls how the maximum context
/// window size is determined:
///
/// * [`CompactPolicy::NoCompact`] — compaction is disabled entirely.
/// * [`CompactPolicy::ModelDefault`] — if `LLMConfig::max_context_tokens` is
///   set it is used; otherwise the client attempts discovery via
///   [`OpenAIClient::discover_max_context_tokens`] (probing `/models` for
///   fields such as `max_model_len`).
/// * [`CompactPolicy::Override(n)`] — uses `n` directly, bypassing config and
///   discovery. Useful for testing with a deliberately small window.
///
/// When a context window is known and the prompt tokens of any turn reach
/// >= 75% of that window, a separate summarization request is issued. The
/// > summarizer is instructed to produce a compact summary targeting roughly
/// > 25% of the window. On success:
///
/// * The summary is stored and will appear in subsequent prompts under the
///   `PREVIOUS SESSIONS SUMMARY` heading.
/// * Old entries are dropped from the in-memory typed history (only the most
///   recent few turns are kept verbatim).
///
/// Compaction is best-effort: if the summarization call fails, a warning is
/// logged and the loop continues with the current history. The compaction chat
/// call itself does not count against `max_iterations`.
///
/// # Arguments
/// * `client` — The [`OpenAIClient`] to use.
/// * `agent` — The agent implementing [`AgentLoop`].
/// * `config` — A [`LoopConfig`] (or builder result) controlling iteration
///   limits and compaction policy.
///
/// # Returns
/// A tuple of `(agent, output)` on success. The agent is returned so
/// implementors can inspect its internal state after the loop completes.
///
/// # Errors
/// Returns an error if the LLM API fails, deserialization fails, or the
/// iteration limit is exceeded (when `max_iterations` is set).
pub async fn run_loop<Agent: AgentLoop + Send>(
    mut client: OpenAIClient,
    mut agent: Agent,
    config: LoopConfig,
) -> anyhow::Result<(Agent, Agent::Output)> {
    let max_iterations = config.max_iterations;
    let compact_policy = config.compact_policy;

    // Determine the effective max context window (in tokens).
    let max_context: Option<usize> = match compact_policy {
        CompactPolicy::NoCompact => {
            info!(target: "da_harness::loop", "compaction disabled by policy");
            None
        }
        CompactPolicy::Override(n) => {
            info!(target: "da_harness::loop", max_context_tokens = n, "using explicit max context (override)");
            Some(n)
        }
        CompactPolicy::ModelDefault => {
            if let Some(n) = client.max_context_tokens() {
                info!(target: "da_harness::loop", max_context_tokens = n, "using configured max context from LLMConfig");
                Some(n)
            } else {
                match client.discover_max_context_tokens().await {
                    Ok(Some(n)) => {
                        info!(target: "da_harness::loop", max_context_tokens = n, "discovered model context window");
                        Some(n)
                    }
                    Ok(None) => {
                        warn!(target: "da_harness::loop", "server did not report a context window for model; compaction disabled");
                        None
                    }
                    Err(e) => {
                        warn!(target: "da_harness::loop", error = %e, "context window discovery failed; compaction disabled");
                        None
                    }
                }
            }
        }
    };

    if max_context.is_none() && compact_policy != CompactPolicy::NoCompact {
        warn!(target: "da_harness::loop", "no max context window known; compaction will not be triggered");
    }

    // Determine iteration range.
    let iter_limit = max_iterations.unwrap_or(usize::MAX);

    // Attempt to restore previously saved state.
    let restored = agent.restore().await;
    let (mut history, mut current_request, mut previous_summary) = match restored {
        Some(state) => {
            info!(target: "da_harness::loop", entries = state.history.len(), has_summary = state.previous_summary.is_some(), "restored saved state");
            (state.history, state.current_request, state.previous_summary)
        }
        None => (Vec::new(), agent.initial_input(), None),
    };

    info!(target: "da_harness::loop", system = %agent.system_prompt(), "starting agent loop");

    // Tracks whether the most recent iteration triggered compaction.
    let mut save_point = SavePoint::Continue;

    for iteration in 0..iter_limit {
        // Persistence save hook (skip iteration 0, which is the initial state).
        if iteration > 0 {
            agent
                .save(
                    &history,
                    previous_summary.as_deref(),
                    &current_request,
                    save_point,
                )
                .await;
        }

        let user_message = build_user_prompt(
            &agent,
            &history,
            &current_request,
            previous_summary.as_deref(),
        );

        info!(target: "da_harness::loop", iteration, prompt = %user_message, ">> sending to LLM");

        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(agent.system_prompt())
                .build()
                .context("building system message")?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_message)
                .build()
                .context("building user message")?
                .into(),
        ];

        let (response_text, prompt_tokens) = client
            .chat_with_usage(messages, 0.6)
            .await
            .context("LLM chat call failed")?;

        info!(target: "da_harness::loop", iteration, reply = %response_text, used_prompt_tokens = ?prompt_tokens, "<< raw LLM reply");

        let json_text = extract_json(&response_text);
        let response: Agent::Response = serde_json::from_str(json_text).context(format!(
            "failed to deserialize LLM response: {}",
            response_text
        ))?;

        history.push((current_request, response.clone()));

        let used = prompt_tokens;

        match agent.iteration(response, &history).await {
            LoopControl::Continue(next_request) => {
                save_point = SavePoint::Continue;
                current_request = next_request;

                // Check for compaction after each response when we know we will continue.
                if let (Some(max), Some(used_tokens)) = (max_context, used)
                    && (used_tokens as usize) >= ((max as f64 * 0.75) as usize)
                {
                    let target = max / 4; // aim for ~25%
                    info!(
                        target: "da_harness::loop",
                        used = used_tokens,
                        max,
                        target,
                        "context usage >= 75% of window; triggering compaction"
                    );

                    match compact_conversation::<Agent>(
                        &client,
                        &history,
                        previous_summary.as_deref(),
                        target,
                    )
                    .await
                    {
                        Ok(summary) => {
                            previous_summary = Some(summary);
                            save_point = SavePoint::Compacted;
                            // Truncate typed history: keep only the most recent few turns.
                            const KEEP_LAST: usize = 3;
                            let len = history.len();
                            if len > KEEP_LAST {
                                history.drain(0..len - KEEP_LAST);
                                info!(
                                    target: "da_harness::loop",
                                    kept = KEEP_LAST,
                                    "history truncated after compaction"
                                );
                            }
                        }
                        Err(e) => {
                            warn!(
                                target: "da_harness::loop",
                                error = %e,
                                "compaction summarization failed; continuing with full history"
                            );
                        }
                    }
                }
            }
            LoopControl::Stop(output) => {
                if let Some((last_req, _)) = history.last() {
                    agent
                        .save(
                            &history,
                            previous_summary.as_deref(),
                            last_req,
                            SavePoint::Final,
                        )
                        .await;
                }
                return Ok((agent, output));
            }
        }
    }

    if let Some(limit) = max_iterations {
        anyhow::bail!("too many iterations (max {})", limit)
    } else {
        anyhow::bail!("agent loop did not terminate; consider setting max_iterations")
    }
}

/// Builds a combined JSON schema containing both the `Request` and `Response`
/// schemas for an agent, with shared type definitions deduplicated under a
/// single `definitions` section.
fn build_combined_schema<Agent: AgentLoop>() -> serde_json::Value {
    let mut schema_gen = SchemaGenerator::default();
    let request_schema = <Agent::Request as schemars::JsonSchema>::json_schema(&mut schema_gen);
    let response_schema = <Agent::Response as schemars::JsonSchema>::json_schema(&mut schema_gen);

    let definitions = schema_gen.take_definitions();

    let mut schema_obj = serde_json::Map::new();
    if !definitions.is_empty() {
        // Use the key name matching schemars' definitions_path setting
        schema_obj.insert(
            "definitions".into(),
            serde_json::to_value(definitions).unwrap(),
        );
    }
    schema_obj.insert(
        "Request".into(),
        serde_json::to_value(request_schema).unwrap(),
    );
    schema_obj.insert(
        "Response".into(),
        serde_json::to_value(response_schema).unwrap(),
    );

    serde_json::Value::Object(schema_obj)
}

/// Builds the full user prompt for a single agent iteration.
///
/// The prompt is assembled from these sections (in order):
/// 1. The agent's user prompt instructions
/// 2. Combined JSON schema for request and response types
/// 3. Few-shot examples (if any)
/// 4. Optional "previous sessions summary" produced by automatic compaction
/// 5. Conversation history (request/response pairs) — may be truncated after compaction
/// 6. Extended context from [`AgentLoop::extend_prompt`] (if provided)
/// 7. The current serialized request as the active input
fn build_user_prompt<Agent: AgentLoop>(
    agent: &Agent,
    history: &[(Agent::Request, Agent::Response)],
    current_request: &Agent::Request,
    previous_summary: Option<&str>,
) -> String {
    let mut buf = String::new();

    buf.push_str(agent.user_prompt());
    buf.push('\n');

    // JSON schema section (shared definitions for both Request and Response)
    let combined_schema = build_combined_schema::<Agent>();
    let schema_json =
        serde_json::to_string_pretty(&combined_schema).expect("serialize combined schema");
    buf.push_str("## JSON SCHEMA\n\n");
    buf.push_str("All INPUT and OUTPUT must conform to these JSON schemas:\n\n");
    buf.push_str("```json\n");
    buf.push_str(&schema_json);
    buf.push_str("\n```\n\n");

    // Examples
    let examples = agent.examples();
    if !examples.is_empty() {
        buf.push_str("## EXAMPLES\n\n");
        for (i, (req, resp)) in examples.iter().enumerate() {
            let req_json = serde_json::to_string(req).expect("serialize example request");
            let resp_json = serde_json::to_string(resp).expect("serialize example response");
            buf.push_str(&format!(
                "Example {}:\nINPUT (JSON): {}\nOUTPUT (JSON): {}\n\n",
                i + 1,
                req_json,
                resp_json,
            ));
        }
    }

    // Previous sessions summary (result of compaction / rewritten history)
    if let Some(summary) = previous_summary {
        let trimmed = summary.trim();
        if !trimmed.is_empty() {
            buf.push_str("## PREVIOUS SESSIONS SUMMARY\n\n");
            buf.push_str(trimmed);
            buf.push_str("\n\n");
        }
    }

    // History pairs (may have been truncated by compaction)
    if !history.is_empty() {
        buf.push_str("## CONVERSATION HISTORY\n\n");
        for (req, resp) in history {
            let req_json = serde_json::to_string(req).expect("serialize history request");
            let resp_json = serde_json::to_string(resp).expect("serialize history response");
            buf.push_str(&format!(
                "INPUT (JSON): {}\nOUTPUT (JSON): {}\n\n",
                req_json, resp_json
            ));
        }
    }

    // Extend prompt hook
    if let Some(extended) = agent.extend_prompt(history) {
        buf.push_str("## ADDITIONAL CONTEXT\n");
        buf.push_str(&extended);
        buf.push('\n');
    }

    // Current input
    let current_json = serde_json::to_string(current_request).expect("serialize current request");
    buf.push_str("## CURRENT INPUT (JSON):\n");
    buf.push_str(&current_json);
    buf.push('\n');

    buf
}

/// Extracts the JSON content from an LLM reply that may be wrapped in
/// markdown code fences (e.g. "```json\n{...}\n```" or "```\n{...}\n```").
///
/// Many local models emit fenced JSON even when instructed to output raw JSON.
/// This helper makes the harness tolerant of that common behavior while
/// preserving the original text for error reporting.
fn extract_json(text: &str) -> &str {
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

/// Performs a separate LLM call to produce a compact summary of the current
/// conversation history (plus any previous summary) that is intended to be
/// small enough to occupy roughly 25% of the model's context window.
///
/// The summary is plain text (not a typed agent turn). On success it is
/// injected by the run loop under `## PREVIOUS SESSIONS SUMMARY` and the
/// typed history is truncated.
async fn compact_conversation<Agent: AgentLoop>(
    client: &OpenAIClient,
    history: &[(Agent::Request, Agent::Response)],
    previous_summary: Option<&str>,
    target_tokens: usize,
) -> anyhow::Result<String> {
    if history.is_empty() && previous_summary.is_none() {
        return Ok(String::new());
    }

    let system = "You are a context compression assistant for a structured JSON agent loop. \
                  Produce a dense, factual memory of everything important that happened so far.";

    let mut user = String::new();
    if let Some(s) = previous_summary {
        user.push_str("Summary of turns before the recent history below:\n");
        user.push_str(s.trim());
        user.push_str("\n\n");
    }

    user.push_str("Recent turns (INPUT/OUTPUT pairs, oldest first):\n\n");
    for (i, (req, resp)) in history.iter().enumerate() {
        let rj = serde_json::to_string(req).unwrap_or_default();
        let sj = serde_json::to_string(resp).unwrap_or_default();
        user.push_str(&format!(
            "Turn {}:\nINPUT: {}\nOUTPUT: {}\n\n",
            i + 1,
            rj,
            sj
        ));
    }

    user.push_str(&format!(
        "Create a single concise summary of the entire conversation (including any prior summary). \
         Preserve goals, key facts, discovered state, decisions made, and what remains to be done. \
         This summary will be re-injected by the controller in future prompts under the heading \
         'PREVIOUS SESSIONS SUMMARY'. Target size: approximately {} tokens (roughly {} characters). \
         Output ONLY the summary text — no explanations, no JSON, no fences.",
        target_tokens,
        target_tokens.saturating_mul(4)
    ));

    let messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system)
            .build()
            .context("building compaction system message")?
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(user)
            .build()
            .context("building compaction user message")?
            .into(),
    ];

    let (mut summary, _usage) = client
        .chat_with_usage(messages, 0.2)
        .await
        .context("compaction chat call failed")?;

    // Best-effort second pass if the first summary is obviously too large
    // (we have no tokenizer in v1; we rely on the instruction + crude length).
    let rough = summary.len() / 4;
    if rough > target_tokens.saturating_mul(2) && !summary.trim().is_empty() {
        let mut stricter = String::new();
        if let Some(s) = previous_summary {
            stricter.push_str("Prior summary:\n");
            stricter.push_str(s.trim());
            stricter.push_str("\n\n");
        }
        stricter.push_str("Recent turns (INPUT/OUTPUT):\n\n");
        for (i, (req, resp)) in history.iter().enumerate() {
            let rj = serde_json::to_string(req).unwrap_or_default();
            let sj = serde_json::to_string(resp).unwrap_or_default();
            stricter.push_str(&format!(
                "Turn {}:\nINPUT: {}\nOUTPUT: {}\n\n",
                i + 1,
                rj,
                sj
            ));
        }
        stricter.push_str(
            "The previous summary was too long. Produce a MUCH shorter version that still \
             contains the essential facts, decisions, and state. Target the requested token budget. \
             Output only the summary text.",
        );

        let messages2: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(system)
                .build()
                .context("building stricter compaction system message")?
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content(stricter)
                .build()
                .context("building stricter compaction user message")?
                .into(),
        ];

        if let Ok((shorter, _)) = client.chat_with_usage(messages2, 0.2).await {
            summary = shorter;
        }
    }

    Ok(summary.trim().to_string())
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
