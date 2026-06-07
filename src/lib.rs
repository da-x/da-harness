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
//! use da_harness::{AgentLoop, LoopControl, OpenAIClient, run_loop};
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
//! let output = run_loop(client, MyAgent).await.unwrap();
//! println!("Result: {}", output);
//! # }
//! ```

use anyhow::Context;
use async_openai::{
    Client,
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
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
}

impl Default for LLMConfig {
    /// Returns a default configuration pointing to a local endpoint at
    /// `http://127.0.0.1:4242/v1` with the model name `LocalModel`.
    fn default() -> Self {
        Self {
            api_base: "http://127.0.0.1:4242/v1".to_owned(),
            model_name: "LocalModel".to_owned(),
            api_key: "".to_owned(),
        }
    }
}

/// A client for OpenAI-compatible chat completion APIs.
///
/// Wraps `async_openai::Client` with convenience methods for health checking
/// and readiness polling.
#[derive(Clone)]
pub struct OpenAIClient {
    client: Client<OpenAIConfig>,
    model_name: String,
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
        }
    }

    /// Returns the model name configured for this client.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Sends a chat completion request and returns the assistant's text response.
    ///
    /// # Arguments
    /// * `messages` — The conversation messages to send.
    /// * `temperature` — Sampling temperature (0.0 to 1.0).
    ///
    /// # Errors
    /// Returns an error if the API call fails or the response contains no content.
    pub async fn chat(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
        temperature: f32,
    ) -> anyhow::Result<String> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model_name())
            .messages(messages)
            .temperature(temperature)
            .build()?;

        let response = self.client.chat().create(request).await?;

        response.choices[0]
            .message
            .content
            .clone()
            .ok_or_else(|| anyhow::anyhow!("LLM returned no content"))
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
    fn extend_prompt(
        &self,
        _history: &[(Self::Request, Self::Response)],
    ) -> Option<String> {
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
}

// ─── run_loop ────────────────────────────────────────────────────────

/// Runs an [`AgentLoop`] with the default maximum of 10 iterations.
///
/// This is a convenience wrapper around [`run_loop_with_max`].
///
/// # Arguments
/// * `client` — The [`OpenAIClient`] to use for chat completions.
/// * `agent` — The agent implementing [`AgentLoop`].
///
/// # Returns
/// The agent's final output value on success.
///
/// # Errors
/// Returns an error if the LLM API fails, response deserialization fails,
/// or the maximum iteration count is exceeded.
pub async fn run_loop<Agent: AgentLoop>(
    client: OpenAIClient,
    agent: Agent,
) -> anyhow::Result<Agent::Output> {
    run_loop_with_max(client, agent, 10).await
}

/// Runs an [`AgentLoop`] with a configurable maximum iteration count.
///
/// Each iteration builds a prompt from the agent's system prompt, user prompt,
/// JSON schemas (with shared type definitions), examples, conversation history,
/// and the current request. The LLM response is deserialized into the agent's
/// `Response` type and passed to [`AgentLoop::iteration`] for control decisions.
///
/// # Arguments
/// * `client` — The [`OpenAIClient`] to use for chat completions.
/// * `agent` — The agent implementing [`AgentLoop`].
/// * `max_iterations` — Maximum number of loop iterations before failing.
///
/// # Returns
/// The agent's final output value on success.
///
/// # Errors
/// Returns an error if the LLM API fails, response deserialization fails,
/// or `max_iterations` is exceeded.
pub async fn run_loop_with_max<Agent: AgentLoop>(
    client: OpenAIClient,
    agent: Agent,
    max_iterations: usize,
) -> anyhow::Result<Agent::Output> {
    let mut history: Vec<(Agent::Request, Agent::Response)> = Vec::new();
    let mut current_request = agent.initial_input();

    info!(target: "da_harness::loop", system = %agent.system_prompt(), "starting agent loop");

    for iteration in 0..max_iterations {
        let user_message = build_user_prompt(&agent, &history, &current_request);

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

        let response_text = client
            .chat(messages, 0.6)
            .await
            .context("LLM chat call failed")?;

        info!(target: "da_harness::loop", iteration, reply = %response_text, "<< raw LLM reply");

        let response: Agent::Response = serde_json::from_str(&response_text)
            .context(format!("failed to deserialize LLM response: {}", response_text))?;

        history.push((current_request, response.clone()));

        match agent.iteration(response, &history).await {
            LoopControl::Continue(next_request) => {
                current_request = next_request;
            }
            LoopControl::Stop(output) => {
                return Ok(output);
            }
        }
    }

    anyhow::bail!("too many iterations (max {})", max_iterations)
}

/// Builds a combined JSON schema containing both the `Request` and `Response`
/// schemas for an agent, with shared type definitions deduplicated under a
/// single `definitions` section.
fn build_combined_schema<Agent: AgentLoop>() -> serde_json::Value {
    let mut schema_gen = SchemaGenerator::default();
    let request_schema =
        <Agent::Request as schemars::JsonSchema>::json_schema(&mut schema_gen);
    let response_schema =
        <Agent::Response as schemars::JsonSchema>::json_schema(&mut schema_gen);

    let definitions = schema_gen.take_definitions();

    let mut schema_obj = serde_json::Map::new();
    if !definitions.is_empty() {
        // Use the key name matching schemars' definitions_path setting
        schema_obj.insert(
            "definitions".into(),
            serde_json::to_value(definitions).unwrap(),
        );
    }
    schema_obj.insert("Request".into(), serde_json::to_value(request_schema).unwrap());
    schema_obj.insert("Response".into(), serde_json::to_value(response_schema).unwrap());

    serde_json::Value::Object(schema_obj)
}

/// Builds the full user prompt for a single agent iteration.
///
/// The prompt is assembled from these sections (in order):
/// 1. The agent's user prompt instructions
/// 2. Combined JSON schema for request and response types
/// 3. Few-shot examples (if any)
/// 4. Conversation history (request/response pairs)
/// 5. Extended context from [`AgentLoop::extend_prompt`] (if provided)
/// 6. The current serialized request as the active input
fn build_user_prompt<Agent: AgentLoop>(
    agent: &Agent,
    history: &[(Agent::Request, Agent::Response)],
    current_request: &Agent::Request,
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

    // History pairs
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

#[cfg(test)]
mod tests {
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
    fn test_shared_type_deduplicated() {
        let mut schema_gen = SchemaGenerator::default();
        let request_schema =
            <TestRequest as schemars::JsonSchema>::json_schema(&mut schema_gen);
        let response_schema =
            <TestResponse as schemars::JsonSchema>::json_schema(&mut schema_gen);

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
        schema_obj.insert("Request".into(), serde_json::to_value(&request_schema).unwrap());
        schema_obj.insert("Response".into(), serde_json::to_value(&response_schema).unwrap());

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
