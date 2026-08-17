//! Multi-tool agent loop: the LLM drives itself by calling named tools.
//!
//! # History-rewriting tools
//!
//! Most tools only return a string result that is appended as a `role=tool`
//! message. **Rewriting tools** (see [`Tool::new_rewriting`]) may also replace
//! the entire conversation history. Typical uses:
//!
//! - Model-directed **self-compaction** (summarize old turns, keep a tail)
//! - Redaction / session branching / other deliberate history edits
//!
//! This is complementary to [`crate::single_tool`]'s *automatic* compaction,
//! which rewrites typed request/response history when token usage is high.
//! Here the **model** chooses when to call a rewrite tool; there is no
//! automatic token-threshold trigger in this module. Per-turn [`TokenUsage`] is
//! forwarded via [`AgentInvocation::usage_callback`] so hosts can log session
//! totals, cost, and context fill.
//!
//! ## Protocol invariants for rewriting handlers
//!
//! When the handler runs, `messages` already includes the assistant message
//! that requested this tool call (and any earlier tool results from the same
//! serial batch). The handler chooses one of two legal outcomes:
//!
//! **Keep this call** (typical self-compaction). The returned list **must**:
//!
//! 1. Be a legal chat prefix for appending one more `role=tool` message with
//!    this call's `tool_call_id`.
//! 2. End with that assistant `tool_calls` turn, **or** with sibling `role=tool`
//!    results from this batch that immediately follow it. After an earlier
//!    sibling has appended its result, last is a `Tool` message — that is
//!    valid. A trailing `User` / `System` is not.
//! 3. **Not** already include the tool response for this call — the loop
//!    appends it after the handler returns.
//!
//! **Consume this call** (the rewrite erases the tool from history). Omit this
//! `tool_call_id` from every assistant `tool_calls` list and do not include a
//! `role=tool` result for it. The loop then does **not** append a tool result.
//! Remaining sibling `tool_calls` on that assistant (and their already-appended
//! results) stay; an assistant left with no `tool_calls` and no content may be
//! dropped, so history may end on a `User` summary.
//!
//! Returning the snapshot unchanged is an **identity rewrite** (e.g. a mark
//! that only produces a tool result). The loop then skips
//! [`AgentInvocation::messages_replace_callback`] and does not swap history.
//!
//! A common self-compaction shape:
//!
//! ```text
//! [system] + [optional summary turn] + [recent tail] + [assistant tool_calls]
//! ```
//!
//! then the loop appends e.g. `tool: "compacted N → M messages"`.
//!
//! ## Parallelism
//!
//! Rewriting needs exclusive ownership of history. If **any** tool call in a
//! batch is a rewriting tool, that batch runs **serially** (even when
//! `parallel_tools` is true). Tool order from the model is preserved: a rewrite
//! sees only tool results already appended earlier in the same batch.
//! Identity-style tools (return the snapshot unchanged) are safe anywhere in
//! the batch. Prefer calling **destructive** rewrite tools alone (or last).
//!
//! ## Persistence
//!
//! - [`AgentInvocation::messages_push_callback`] — append-only mirror of each
//!   new message.
//! - [`AgentInvocation::messages_replace_callback`] — full replace after a
//!   rewriting tool, with `(old_messages, new_messages)`; hosts that persist
//!   history should treat the new list as authoritative (do not rely on push
//!   alone after a rewrite).
//!
//! ## Testing without a live LLM
//!
//! Set [`AgentInvocation::inference_callback`] to replace each `chat_with_tools`
//! turn with a deterministic callback, then call
//! [`AgentInvocation::run_without_client`]. Helpers [`assistant_text`] and
//! [`assistant_tool_calls`] build mock model responses for tests.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestToolMessageArgs, ChatCompletionRequestUserMessageArgs,
    ChatCompletionRequestUserMessageContent, ChatCompletionResponseMessage, ChatCompletionTool,
    Role,
};
use futures::{FutureExt, future::BoxFuture};
use serde::Deserialize;
use tokio_retry::Retry;
use tracing::{debug, info, warn};

use crate::{OpenAIClient, TokenUsage, generate_tool_schema};

pub use tokio_retry;

pub type TaskFuture = BoxFuture<'static, anyhow::Result<()>>;
pub type TaskFutureStr = BoxFuture<'static, anyhow::Result<String>>;

/// Callback after a single message is appended to conversation history.
pub type MessagesPushCallback = Arc<dyn Fn(&ChatCompletionRequestMessage) + Send + Sync>;

/// Callback after a rewriting tool replaces the full conversation history.
///
/// Arguments are `(old_messages, new_messages)` — the history before the rewrite
/// and the replacement list returned by the tool (before this call's tool result
/// is appended).
pub type MessagesReplaceCallback =
    Arc<dyn Fn(&[ChatCompletionRequestMessage], &[ChatCompletionRequestMessage]) + Send + Sync>;

/// Callback after a successful LLM chat turn that reported token usage.
///
/// Not invoked when the server omits usage, or when [`InferenceCallback`] is used
/// (mock turns have no usage).
pub type UsageCallback = Arc<dyn Fn(TokenUsage) + Send + Sync>;

/// Replaces a single `chat_with_tools` turn when set (primarily for tests).
///
/// Receives a snapshot of the conversation history at inference time.
/// Returns the assistant turn the real model would have produced
/// (`content` and/or `tool_calls`). Token usage is not modeled; the loop
/// sees `usage = None`.
///
/// Use with [`AgentInvocation::run_without_client`]. Helpers
/// [`assistant_text`] and [`assistant_tool_calls`] build responses without
/// depending on `async-openai` field layout.
pub type InferenceCallback = Arc<
    dyn Fn(
            Vec<ChatCompletionRequestMessage>,
        ) -> BoxFuture<'static, anyhow::Result<ChatCompletionResponseMessage>>
        + Send
        + Sync,
>;

/// Build a text-only assistant response for use with [`InferenceCallback`].
pub fn assistant_text(content: impl Into<String>) -> ChatCompletionResponseMessage {
    #[allow(deprecated)]
    ChatCompletionResponseMessage {
        content: Some(content.into()),
        refusal: None,
        tool_calls: None,
        role: Role::Assistant,
        function_call: None,
        audio: None,
    }
}

/// Build an assistant response that requests tool calls for use with
/// [`InferenceCallback`].
pub fn assistant_tool_calls(
    calls: Vec<ChatCompletionMessageToolCall>,
) -> ChatCompletionResponseMessage {
    #[allow(deprecated)]
    ChatCompletionResponseMessage {
        content: None,
        refusal: None,
        tool_calls: Some(calls),
        role: Role::Assistant,
        function_call: None,
        audio: None,
    }
}

/// Result of a history-rewriting tool: tool message content + replacement history.
pub type RewriteOutcome = (String, Vec<ChatCompletionRequestMessage>);

/// Handler for a history-rewriting tool.
///
/// Receives parsed JSON args and an owned snapshot of the current conversation
/// (including the assistant `tool_calls` turn that invoked this tool). Returns
/// `(tool_result_content, new_messages)`.
///
/// See the [module-level docs](self) for protocol invariants.
pub type RewritingHandler = Arc<
    dyn Fn(
            serde_json::Value,
            Vec<ChatCompletionRequestMessage>,
        ) -> BoxFuture<'static, anyhow::Result<RewriteOutcome>>
        + Send
        + Sync,
>;

/// Typed callback for [`Tool::new_rewriting`]: `(args, messages) → (content, new_messages)`.
pub type TypedRewritingCallback<T> = Arc<
    dyn Fn(
            T,
            Vec<ChatCompletionRequestMessage>,
        ) -> BoxFuture<'static, anyhow::Result<RewriteOutcome>>
        + Send
        + Sync,
>;

/// Factory that produces a fresh [tokio-retry](https://crates.io/crates/tokio-retry) strategy
/// (an iterator of inter-attempt delays) for each LLM call.
///
/// Prefer setting this via [`AgentInvocationArgs::retry_strategy`], which accepts any
/// cloneable strategy such as `ExponentialBackoff::from_millis(100).map(jitter).take(5)`.
pub type RetryStrategyFactory =
    Arc<dyn Fn() -> Box<dyn Iterator<Item = Duration> + Send> + Send + Sync>;

/// User request to the agent loop
pub enum UserRequest {
    /// Append a user message to the session
    Message(ChatCompletionRequestUserMessageContent),

    /// Change temperature used with the LLM
    ChangeTemperature(f32),

    /// Exit the loop
    Exit,
}

fn default_message_callback() -> Arc<dyn Fn(String) -> TaskFuture + Send + Sync> {
    Arc::new(|_: String| async move { Ok(()) }.boxed())
}

fn default_idle_callback() -> Arc<dyn Fn() -> TaskFuture + Send + Sync> {
    Arc::new(|| async move { Ok(()) }.boxed())
}

fn default_push_callback() -> MessagesPushCallback {
    Arc::new(|_: &ChatCompletionRequestMessage| {})
}

fn default_replace_callback() -> MessagesReplaceCallback {
    Arc::new(|_: &[ChatCompletionRequestMessage], _: &[ChatCompletionRequestMessage]| {})
}

fn default_usage_callback() -> UsageCallback {
    Arc::new(|_: TokenUsage| {})
}

/// Internal tool implementation: normal result-only vs history-rewriting.
#[derive(Clone)]
enum ToolBody {
    /// Args → result string. Conversation history is unchanged.
    Normal(Arc<dyn Fn(serde_json::Value) -> TaskFutureStr + Send + Sync>),
    /// May replace the entire message list; see [`Tool::new_rewriting`].
    Rewriting(RewritingHandler),
}

/// A tool the multi-tool agent can invoke.
///
/// Construct with [`Tool::new`] for ordinary tools, or [`Tool::new_rewriting`]
/// when the tool is allowed to rewrite chat history (e.g. self-compaction).
#[derive(Clone)]
pub struct Tool {
    body: ToolBody,
    description: ChatCompletionTool,
}

impl Tool {
    /// Create a normal tool from a typed async callback.
    ///
    /// Argument type `T` must deserialize from the model's JSON and implement
    /// [`schemars::JsonSchema`] so the OpenAI tool schema can be generated.
    pub fn new<T>(callback: Arc<dyn Fn(T) -> TaskFutureStr + Send + Sync>) -> anyhow::Result<Self>
    where
        T: for<'de> Deserialize<'de> + Clone + schemars::JsonSchema + 'static,
    {
        let callback = callback;
        let handler: Arc<dyn Fn(serde_json::Value) -> TaskFutureStr + Send + Sync> =
            Arc::new(move |value: serde_json::Value| {
                let cb = callback.clone();
                async move { cb(serde_json::from_value(value).expect("valid tool args")).await }
                    .boxed()
            });

        let description = generate_tool_schema::<T>()?;

        Ok(Self {
            body: ToolBody::Normal(handler),
            description,
        })
    }

    /// Create a **history-rewriting** tool from a typed async callback.
    ///
    /// The callback receives `(args, messages_snapshot)` and returns
    /// `(tool_content, new_messages)`. The loop replaces in-memory history with
    /// `new_messages` and fires [`AgentInvocation::messages_replace_callback`].
    /// If the replacement still lists this call's `tool_call_id`, the loop then
    /// appends a normal `role=tool` message with `tool_content`. If the handler
    /// **consumed** the call (id absent from every assistant `tool_calls` list),
    /// no result is appended and `tool_content` is only logged.
    ///
    /// Handlers that need async work (e.g. an LLM summarization call) can freely
    /// `.await` because they own the message snapshot — see module docs for why
    /// this is not `&mut Vec`.
    ///
    /// # Protocol
    ///
    /// Either keep this call and return a valid prefix for appending its
    /// result, or omit the call id (and any result for it) to consume the call.
    /// Returning the snapshot unchanged is an identity rewrite (no replace
    /// callback). See the [module-level docs](self).
    pub fn new_rewriting<T>(callback: TypedRewritingCallback<T>) -> anyhow::Result<Self>
    where
        T: for<'de> Deserialize<'de> + Clone + schemars::JsonSchema + 'static,
    {
        let handler: RewritingHandler = Arc::new(move |value, messages| {
            let cb = callback.clone();
            async move {
                let args: T = serde_json::from_value(value).expect("valid tool args");
                cb(args, messages).await
            }
            .boxed()
        });

        let description = generate_tool_schema::<T>()?;

        Ok(Self {
            body: ToolBody::Rewriting(handler),
            description,
        })
    }

    /// OpenAI function name for this tool (from the generated schema).
    pub fn name(&self) -> &str {
        self.description.function.name.as_str()
    }

    /// Whether this tool may replace conversation history when invoked.
    pub fn is_rewriting(&self) -> bool {
        matches!(self.body, ToolBody::Rewriting(_))
    }
}

#[derive(derive_builder::Builder)]
#[builder(name = "AgentInvocationArgs")]
#[builder(pattern = "owned")]
#[builder(setter(into))]
pub struct AgentInvocation {
    pub system_prompt: String,

    /// Tools the agent can invoke. The tool's function name is used to differentiate
    /// when the LLM tells us to execute something, and to invoke the correct handler
    /// as an async task. If we receive a `tool_calls` vector of multiple items we
    /// issue the tasks in parallel or serially based on `parallel_tools` (rewriting
    /// tools always force serial execution for that batch — see module docs).
    #[builder(default)]
    pub tools: Vec<Tool>,

    #[builder(default = "false")]
    pub parallel_tools: bool,

    /// Process a request to the session
    pub incoming: tokio::sync::mpsc::Receiver<UserRequest>,

    /// Called when agent produces a text response (no tool calls).
    #[builder(default = "default_message_callback()")]
    pub agent_message_callback: Arc<dyn Fn(String) -> TaskFuture + Send + Sync>,

    /// Called when the agent is idle, waiting for incoming user messages.
    #[builder(default = "default_idle_callback()")]
    pub agent_idle_callback: Arc<dyn Fn() -> TaskFuture + Send + Sync>,

    /// Called after a message is pushed to the conversation history.
    #[builder(default = "default_push_callback()")]
    pub messages_push_callback: MessagesPushCallback,

    /// Called after a rewriting tool replaces the in-memory history, with
    /// `(old_messages, new_messages)` — before the tool result for that call is
    /// appended.
    ///
    /// Hosts that mirror history for persistence should treat `new_messages` as
    /// authoritative: clear the stored list and store these messages, then
    /// continue applying [`Self::messages_push_callback`] for subsequent appends.
    /// `old_messages` is useful for diffing, audit logs, or selective retention.
    #[builder(default = "default_replace_callback()")]
    pub messages_replace_callback: MessagesReplaceCallback,

    /// Called after each successful LLM turn that reports token usage.
    ///
    /// Hosts can accumulate session totals, estimate cost, and log context fill.
    /// Not called for mock inference or when the server omits usage.
    #[builder(default = "default_usage_callback()")]
    pub usage_callback: UsageCallback,

    /// Optional [tokio-retry](https://crates.io/crates/tokio-retry) strategy for each
    /// `chat_with_tools` LLM call.
    ///
    /// When set, a fresh strategy is produced for every request and transient failures
    /// are retried with the strategy's delays between attempts. Defaults to no retries
    /// (a single attempt).
    ///
    /// Use [`AgentInvocationArgs::retry_strategy`] to set this from any cloneable
    /// `IntoIterator<Item = Duration>` (e.g. `ExponentialBackoff`, `FixedInterval`).
    ///
    /// Not used when [`Self::inference_callback`] is set.
    #[builder(default, setter(custom))]
    pub retry_strategy: Option<RetryStrategyFactory>,

    /// When set, used instead of [`OpenAIClient::chat_with_tools`] for each
    /// inference turn. Intended for deterministic harness tests.
    ///
    /// Prefer [`AgentInvocation::run_without_client`] so no live client is required.
    /// If both a client and this callback are provided (via [`AgentInvocation::run`]),
    /// the callback wins and no network call is made.
    #[builder(default, setter(strip_option))]
    pub inference_callback: Option<InferenceCallback>,

    /// Optional conversation history inserted **after** the system prompt and
    /// **before** the loop accepts new [`UserRequest`] messages.
    ///
    /// Hosts use this for session continuation (e.g. replaying prior turns).
    /// Each message is pushed via [`Self::messages_push_callback`]. Seed-only
    /// history does not trigger an LLM turn by itself — the loop still waits
    /// for an incoming user message (or a prior tool-call continuation path).
    ///
    /// Callers should supply a legal chat prefix (typically user / assistant /
    /// tool turns). Do not include a system message here; the live
    /// [`Self::system_prompt`] is always message 0.
    #[builder(default)]
    pub seed_messages: Vec<ChatCompletionRequestMessage>,
}

impl AgentInvocationArgs {
    /// Set a [tokio-retry](https://crates.io/crates/tokio-retry) strategy for each
    /// `chat_with_tools` call.
    ///
    /// The strategy is cloned for every LLM request. Any cloneable
    /// `IntoIterator<Item = Duration>` works, including the strategies from
    /// `tokio_retry::strategy` and adapters such as `.map(jitter).take(n)`:
    ///
    /// ```ignore
    /// use da_harness::multi_tool::tokio_retry::strategy::{ExponentialBackoff, jitter};
    ///
    /// AgentInvocationArgs::default()
    ///     .retry_strategy(
    ///         ExponentialBackoff::from_millis(100)
    ///             .map(jitter)
    ///             .take(5),
    ///     )
    /// ```
    pub fn retry_strategy<S, I>(mut self, strategy: S) -> Self
    where
        S: IntoIterator<IntoIter = I, Item = Duration> + Clone + Send + Sync + 'static,
        I: Iterator<Item = Duration> + Send + 'static,
    {
        let factory: RetryStrategyFactory =
            Arc::new(move || Box::new(strategy.clone().into_iter()));
        self.retry_strategy = Some(Some(factory));
        self
    }
}

/// Whether any assistant turn still lists `tool_call_id` in `tool_calls`.
fn history_has_tool_call_id(messages: &[ChatCompletionRequestMessage], tool_call_id: &str) -> bool {
    messages.iter().any(|m| match m {
        ChatCompletionRequestMessage::Assistant(asst) => asst
            .tool_calls
            .as_ref()
            .is_some_and(|cs| cs.iter().any(|c| c.id == tool_call_id)),
        _ => false,
    })
}

/// Whether any `role=tool` message is a result for `tool_call_id`.
fn history_has_tool_result_id(
    messages: &[ChatCompletionRequestMessage],
    tool_call_id: &str,
) -> bool {
    messages.iter().any(|m| match m {
        ChatCompletionRequestMessage::Tool(t) => t.tool_call_id == tool_call_id,
        _ => false,
    })
}

/// Whether `messages` can take one more `role=tool` for `tool_call_id`.
///
/// Walks backward over any trailing sibling `Tool` results; the first
/// non-tool message must be an `Assistant`. If that turn lists `tool_calls`,
/// it must include `tool_call_id`.
fn can_append_tool_result(messages: &[ChatCompletionRequestMessage], tool_call_id: &str) -> bool {
    for msg in messages.iter().rev() {
        match msg {
            ChatCompletionRequestMessage::Tool(_) => continue,
            ChatCompletionRequestMessage::Assistant(asst) => {
                return match &asst.tool_calls {
                    None => true,
                    Some(calls) => calls.iter().any(|c| c.id == tool_call_id),
                };
            }
            _ => return false,
        }
    }
    false
}

impl AgentInvocation {
    /// Run the agent loop against a live OpenAI-compatible client.
    ///
    /// If [`Self::inference_callback`] is set, that callback is used for each
    /// inference turn and the client is not contacted.
    pub async fn run(self, client: OpenAIClient) -> anyhow::Result<()> {
        self.run_inner(Some(client)).await
    }

    /// Run the agent loop without a live LLM client.
    ///
    /// Requires [`Self::inference_callback`] to be set; each inference turn is
    /// answered by that callback instead of `chat_with_tools`.
    pub async fn run_without_client(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.inference_callback.is_some(),
            "run_without_client requires inference_callback"
        );
        self.run_inner(None).await
    }

    async fn run_inner(mut self, client: Option<OpenAIClient>) -> anyhow::Result<()> {
        let tools: Vec<ChatCompletionTool> =
            self.tools.iter().map(|t| t.description.clone()).collect();

        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content(self.system_prompt.as_str())
                .build()
                .context("building system message")?
                .into(),
        ];
        debug!(target: "da_harness::multi_tool", role = "system", msg = %&self.system_prompt);

        // Session-continuation seed: after system, before any new user input.
        let seed_n = self.seed_messages.len();
        if seed_n > 0 {
            info!(
                target: "da_harness::multi_tool",
                seed_messages = seed_n,
                "seeding conversation history after system prompt"
            );
            for msg in self.seed_messages.drain(..) {
                messages.push(msg);
                (self.messages_push_callback)(messages.last().unwrap());
            }
        }

        let mut prev_call = false;

        let mut new_user_messages = 0;

        let mut temperature: f32 = 0.5;

        // Track whether we have a pending user message to inject.
        loop {
            // If channel is closed and no pending work, we're done.
            if self.incoming.is_closed() {
                info!(target: "da_harness::multi_tool", "incoming channel closed, ending loop");
                break;
            }

            // Drain any available requests from the incoming channel.
            while let Ok(req) = self.incoming.try_recv() {
                match req {
                    UserRequest::Message(msg) => {
                        let s = serde_json::to_string(&msg)?;
                        debug!(target: "da_harness::multi_tool", msg = s, "<< user message received");
                        messages.push(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(msg)
                                .build()
                                .context("building user message")?
                                .into(),
                        );
                        (self.messages_push_callback)(messages.last().unwrap());
                        new_user_messages += 1;
                    }
                    UserRequest::ChangeTemperature(temp) => {
                        info!(target: "da_harness::multi_tool", temperature = temp, "<< temperature changed");
                        temperature = temp;
                    }
                    UserRequest::Exit => {
                        info!(target: "da_harness::multi_tool", "<< exit requested");
                        return Ok(());
                    }
                }
            }

            let (response, usage) = if prev_call || new_user_messages > 0 {
                new_user_messages = 0;
                self.chat_with_tools_retrying(
                    client.as_ref(),
                    messages.clone(),
                    tools.clone(),
                    temperature,
                )
                .await
                .context("LLM chat call failed")?
            } else {
                (self.agent_idle_callback)().await?;

                if let Some(req) = self.incoming.recv().await {
                    match req {
                        UserRequest::Message(msg) => {
                            let s = serde_json::to_string(&msg)?;
                            debug!(target: "da_harness::multi_tool", msg = s, "<< user message received");
                            messages.push(
                                ChatCompletionRequestUserMessageArgs::default()
                                    .content(msg)
                                    .build()
                                    .context("building user message")?
                                    .into(),
                            );
                            (self.messages_push_callback)(messages.last().unwrap());
                            new_user_messages += 1;
                            continue;
                        }
                        UserRequest::ChangeTemperature(temp) => {
                            info!(target: "da_harness::multi_tool", temperature = temp, "<< temperature changed");
                            temperature = temp;
                            continue;
                        }
                        UserRequest::Exit => {
                            info!(target: "da_harness::multi_tool", "<< exit requested");
                            return Ok(());
                        }
                    }
                } else {
                    break;
                }
            };

            if let Some(u) = usage {
                (self.usage_callback)(u);
            }

            let mut asst_builder = ChatCompletionRequestAssistantMessageArgs::default();
            if let Some(tool_calls) = &response.tool_calls {
                asst_builder.tool_calls(tool_calls.clone());
                prev_call = true;
            } else {
                prev_call = false;
            };

            if let Some(ref content) = response.content {
                info!(target: "da_harness::multi_tool", reply = %content, "<< LLM text response");
                asst_builder.content(content.as_str());
            }
            if let Some(ref c) = response.refusal {
                asst_builder.refusal(c.as_str());
            }

            messages.push(
                asst_builder
                    .build()
                    .context("building tool message")?
                    .into(),
            );
            (self.messages_push_callback)(messages.last().unwrap());

            if let Some(tool_calls) = &response.tool_calls {
                info!(
                    target: "da_harness::multi_tool",
                    n_tools = tool_calls.len(),
                    "<< LLM requested tool calls"
                );

                for tc in tool_calls {
                    let desc = serde_json::to_string(&tc.function)?;
                    debug!(target: "da_harness::multi_tool", role = "tool-call", tool_call_id = %tc.id, desc = %desc);
                }

                // Rewriting tools need exclusive access to history → serial batch.
                let batch_has_rewrite = tool_calls.iter().any(|tc| {
                    self.tools
                        .iter()
                        .find(|t| t.description.function.name.as_str() == tc.function.name)
                        .is_some_and(|t| t.is_rewriting())
                });

                if self.parallel_tools && !batch_has_rewrite {
                    let futures: Vec<_> =
                        tool_calls.iter().map(|tc| self.execute_tool(tc)).collect();
                    let results = futures::future::join_all(futures).await;

                    for (tc, result) in tool_calls.iter().zip(results) {
                        let content = match result {
                            Ok(s) => s,
                            Err(e) => format!("Error: {}", e),
                        };
                        self.push_tool_result(&mut messages, tc, content)?;
                    }
                } else {
                    if batch_has_rewrite && self.parallel_tools {
                        debug!(
                            target: "da_harness::multi_tool",
                            "batch includes a rewriting tool; forcing serial execution"
                        );
                    }
                    // Serial path: normal tools and/or rewriting tools in model order.
                    // After a rewrite that *keeps* this call, `messages` is a prefix
                    // and we append this call's tool result. A rewrite that *consumed*
                    // the call (id no longer in any assistant `tool_calls`) is final.
                    for tc in tool_calls {
                        let content = match self.execute_tool_serial(tc, &mut messages).await {
                            Ok(s) => s,
                            Err(e) => format!("Error: {}", e),
                        };
                        if history_has_tool_call_id(&messages, &tc.id) {
                            self.push_tool_result(&mut messages, tc, content)?;
                        } else {
                            debug!(
                                target: "da_harness::multi_tool",
                                tool_call_id = %tc.id,
                                "rewriting tool consumed its own call; skipping tool result"
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Append a `role=tool` message and notify the push callback.
    fn push_tool_result(
        &self,
        messages: &mut Vec<ChatCompletionRequestMessage>,
        tc: &async_openai::types::ChatCompletionMessageToolCall,
        content: String,
    ) -> anyhow::Result<()> {
        debug!(target: "da_harness::multi_tool", role = "tool-rsp", tool_call_id = %tc.id, content = %content);
        messages.push(
            ChatCompletionRequestToolMessageArgs::default()
                .tool_call_id(tc.id.clone())
                .content(content)
                .build()
                .context("building tool message")?
                .into(),
        );
        (self.messages_push_callback)(messages.last().unwrap());
        Ok(())
    }

    /// Execute a normal (non-rewriting) tool. Used for parallel batches.
    ///
    /// Rewriting tools must not be scheduled on this path; the run loop forces
    /// serial execution when any call in the batch is rewriting.
    fn execute_tool(
        &self,
        tool_call: &async_openai::types::ChatCompletionMessageToolCall,
    ) -> BoxFuture<'static, anyhow::Result<String>> {
        let tools = self.tools.clone();
        let name = tool_call.function.name.clone();
        let args = tool_call.function.arguments.clone();

        async move {
            let tool = tools
                .iter()
                .find(|t| t.description.function.name.as_str() == name)
                .ok_or_else(|| anyhow::anyhow!("unknown tool: {}", name))?;

            let parsed: serde_json::Value =
                serde_json::from_str(&args).context("failed to parse tool arguments")?;

            match &tool.body {
                ToolBody::Normal(handler) => handler(parsed).await,
                ToolBody::Rewriting(_) => anyhow::bail!(
                    "internal error: rewriting tool '{}' invoked on parallel execute_tool path",
                    name
                ),
            }
        }
        .boxed()
    }

    /// Execute one tool with access to live history (serial path).
    ///
    /// - **Normal:** same as [`Self::execute_tool`].
    /// - **Rewriting:** snapshot → handler → replace `messages` → replace callback.
    ///   Does **not** append the tool result; the caller does that next when
    ///   this call's id is still present in history.
    async fn execute_tool_serial(
        &self,
        tool_call: &async_openai::types::ChatCompletionMessageToolCall,
        messages: &mut Vec<ChatCompletionRequestMessage>,
    ) -> anyhow::Result<String> {
        let name = tool_call.function.name.as_str();
        let tool = self
            .tools
            .iter()
            .find(|t| t.description.function.name.as_str() == name)
            .ok_or_else(|| anyhow::anyhow!("unknown tool: {}", name))?;

        let parsed: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)
            .context("failed to parse tool arguments")?;

        match &tool.body {
            ToolBody::Normal(handler) => handler(parsed).await,
            ToolBody::Rewriting(handler) => {
                let before_len = messages.len();
                // Snapshot (owned) so the handler may `.await` freely.
                let snapshot = messages.clone();
                let (content, new_messages) = handler(parsed, snapshot).await?;

                anyhow::ensure!(
                    !new_messages.is_empty(),
                    "rewriting tool '{}' returned an empty message list",
                    name
                );

                // Identity: handler returned the snapshot unchanged (e.g. Folding::Mark).
                // Sibling tool results from this batch may already follow the assistant
                // turn; do not treat that as a rewrite and do not fire replace.
                if new_messages == *messages {
                    return Ok(content);
                }

                if history_has_tool_call_id(&new_messages, &tool_call.id) {
                    anyhow::ensure!(
                        can_append_tool_result(&new_messages, &tool_call.id),
                        "rewriting tool '{}' must return a legal prefix for appending \
                         this tool result (assistant tool_calls turn, optionally \
                         followed by sibling role=tool messages)",
                        name
                    );
                } else {
                    anyhow::ensure!(
                        !history_has_tool_result_id(&new_messages, &tool_call.id),
                        "rewriting tool '{}' removed its tool_call but left a \
                         role=tool result for {}",
                        name,
                        tool_call.id
                    );
                }

                info!(
                    target: "da_harness::multi_tool",
                    tool = name,
                    before = before_len,
                    after = new_messages.len(),
                    "history rewritten by tool"
                );

                // Notify before swap so both old and new slices are available.
                // Replace is authoritative for persistence mirrors; push is not
                // replayed for each historical message.
                (self.messages_replace_callback)(messages, &new_messages);
                *messages = new_messages;
                Ok(content)
            }
        }
    }

    /// Call `chat_with_tools` (or the inference callback), optionally retrying.
    ///
    /// When [`Self::inference_callback`] is set, the callback answers the turn
    /// and `client` / retry strategy are unused. Otherwise a live client is required.
    async fn chat_with_tools_retrying(
        &self,
        client: Option<&OpenAIClient>,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Vec<ChatCompletionTool>,
        temperature: f32,
    ) -> anyhow::Result<(ChatCompletionResponseMessage, Option<TokenUsage>)> {
        if let Some(cb) = &self.inference_callback {
            let response = cb(messages).await.context("inference callback failed")?;
            return Ok((response, None));
        }

        let client = client.context("OpenAIClient required when inference_callback is not set")?;
        let parallel_tools = self.parallel_tools;

        match &self.retry_strategy {
            None => {
                client
                    .chat_with_tools(messages, tools, parallel_tools, temperature)
                    .await
            }
            Some(make_strategy) => {
                let client = client.clone();
                let mut attempt: u32 = 0;
                Retry::start(make_strategy(), || {
                    attempt += 1;
                    if attempt > 1 {
                        warn!(
                            target: "da_harness::multi_tool",
                            attempt,
                            "retrying chat_with_tools after failure"
                        );
                    }
                    client.chat_with_tools(
                        messages.clone(),
                        tools.clone(),
                        parallel_tools,
                        temperature,
                    )
                })
                .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionToolType, FunctionCall,
    };

    /// Minimal tool schema args for unit tests.
    #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
    struct NoArgs {}

    /// Identity rewriting tool (returns the snapshot unchanged).
    #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
    struct Mark {}

    #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
    struct KeepLast {
        /// Number of trailing messages to keep after the system message.
        n: usize,
    }

    fn dummy_tool_call(name: &str, args: &str) -> ChatCompletionMessageToolCall {
        dummy_tool_call_id("call_test", name, args)
    }

    fn dummy_tool_call_id(id: &str, name: &str, args: &str) -> ChatCompletionMessageToolCall {
        ChatCompletionMessageToolCall {
            id: id.into(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
        }
    }

    fn tool_result_msg(call_id: &str, content: &str) -> ChatCompletionRequestMessage {
        ChatCompletionRequestToolMessageArgs::default()
            .tool_call_id(call_id)
            .content(content)
            .build()
            .unwrap()
            .into()
    }

    #[tokio::test]
    async fn rewriting_tool_replaces_history_and_returns_content() {
        let tool = Tool::new_rewriting(Arc::new(|args: KeepLast, mut messages| {
            async move {
                // Keep system + last `n` messages (clamped).
                if messages.len() > 1 {
                    let system = messages[0].clone();
                    let tail_start = messages.len().saturating_sub(args.n).max(1);
                    let mut kept = vec![system];
                    kept.extend(messages.drain(tail_start..));
                    messages = kept;
                }
                let content = format!("kept {} messages", messages.len());
                Ok((content, messages))
            }
            .boxed()
        }))
        .unwrap();

        assert!(tool.is_rewriting());
        assert_eq!(tool.name(), "KeepLast");

        let replaced = Arc::new(std::sync::Mutex::new(None::<(usize, usize)>));
        let replaced_cb = replaced.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(tx); // unused; we only exercise execute_tool_serial

        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .tools(vec![tool])
            .incoming(rx)
            .messages_replace_callback({
                let cb: MessagesReplaceCallback = Arc::new(
                    move |old: &[ChatCompletionRequestMessage],
                          new: &[ChatCompletionRequestMessage]| {
                        *replaced_cb.lock().unwrap() = Some((old.len(), new.len()));
                    },
                );
                cb
            })
            .build()
            .unwrap();

        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("old1")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("old2")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![dummy_tool_call("KeepLast", r#"{"n":1}"#)])
                .build()
                .unwrap()
                .into(),
        ];
        let before = messages.len();
        assert_eq!(before, 4);

        let tc = dummy_tool_call("KeepLast", r#"{"n":1}"#);
        let content = invocation
            .execute_tool_serial(&tc, &mut messages)
            .await
            .unwrap();

        // system + last 1 message
        assert_eq!(messages.len(), 2);
        assert!(content.contains("kept 2"));
        assert_eq!(*replaced.lock().unwrap(), Some((before, 2)));
    }

    fn invocation_with_rewrite_tool(
        tool: Tool,
        replaced: Arc<std::sync::Mutex<bool>>,
    ) -> AgentInvocation {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(tx);
        AgentInvocationArgs::default()
            .system_prompt("sys")
            .tools(vec![tool])
            .incoming(rx)
            .messages_replace_callback({
                let cb: MessagesReplaceCallback = Arc::new(move |_, _| {
                    *replaced.lock().unwrap() = true;
                });
                cb
            })
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn identity_rewrite_after_sibling_tool_result_is_noop() {
        let tool = Tool::new_rewriting(Arc::new(|_: Mark, messages| {
            async move { Ok(("OK mark".to_owned(), messages)) }.boxed()
        }))
        .unwrap();

        let replaced = Arc::new(std::sync::Mutex::new(false));
        let invocation = invocation_with_rewrite_tool(tool, replaced.clone());

        let sibling = dummy_tool_call_id("call_todo", "Todo", "{}");
        let mark = dummy_tool_call_id("call_mark", "Mark", "{}");
        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("go")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![sibling, mark.clone()])
                .build()
                .unwrap()
                .into(),
            tool_result_msg("call_todo", "todo-ok"),
        ];
        let before = messages.clone();

        let content = invocation
            .execute_tool_serial(&mark, &mut messages)
            .await
            .unwrap();

        assert_eq!(content, "OK mark");
        assert_eq!(messages, before);
        assert!(
            !*replaced.lock().unwrap(),
            "identity rewrite must not fire replace callback"
        );
        assert!(matches!(
            messages.last(),
            Some(ChatCompletionRequestMessage::Tool(_))
        ));
    }

    #[tokio::test]
    async fn identity_rewrite_when_last_is_assistant_is_noop() {
        let tool = Tool::new_rewriting(Arc::new(|_: Mark, messages| {
            async move { Ok(("OK mark".to_owned(), messages)) }.boxed()
        }))
        .unwrap();

        let replaced = Arc::new(std::sync::Mutex::new(false));
        let invocation = invocation_with_rewrite_tool(tool, replaced.clone());

        let mark = dummy_tool_call_id("call_mark", "Mark", "{}");
        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("go")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![mark.clone()])
                .build()
                .unwrap()
                .into(),
        ];

        let content = invocation
            .execute_tool_serial(&mark, &mut messages)
            .await
            .unwrap();

        assert_eq!(content, "OK mark");
        assert!(!*replaced.lock().unwrap());
        assert!(matches!(
            messages.last(),
            Some(ChatCompletionRequestMessage::Assistant(_))
        ));
    }

    #[tokio::test]
    async fn rewriting_tool_ending_with_user_is_rejected() {
        let tool = Tool::new_rewriting(Arc::new(|_: NoArgs, mut messages| {
            async move {
                messages.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content("summary")
                        .build()
                        .unwrap()
                        .into(),
                );
                Ok(("folded".to_owned(), messages))
            }
            .boxed()
        }))
        .unwrap();

        let replaced = Arc::new(std::sync::Mutex::new(false));
        let invocation = invocation_with_rewrite_tool(tool, replaced.clone());

        let tc = dummy_tool_call("NoArgs", "{}");
        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![tc.clone()])
                .build()
                .unwrap()
                .into(),
        ];

        let err = invocation
            .execute_tool_serial(&tc, &mut messages)
            .await
            .expect_err("user-trailing rewrite must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("legal prefix"), "unexpected error: {msg}");
        assert!(!*replaced.lock().unwrap());
        assert!(matches!(
            messages.last(),
            Some(ChatCompletionRequestMessage::Assistant(_))
        ));
    }

    /// Drop assistant `tool_calls` named `name` and any matching `role=tool` results.
    fn strip_named_tool_calls(messages: &mut Vec<ChatCompletionRequestMessage>, name: &str) {
        let mut drop_ids = std::collections::HashSet::new();
        for m in messages.iter_mut() {
            if let ChatCompletionRequestMessage::Assistant(asst) = m {
                if let Some(calls) = asst.tool_calls.as_mut() {
                    calls.retain(|c| {
                        if c.function.name == name {
                            drop_ids.insert(c.id.clone());
                            false
                        } else {
                            true
                        }
                    });
                    if calls.is_empty() {
                        asst.tool_calls = None;
                    }
                }
            }
        }
        messages.retain(|m| match m {
            ChatCompletionRequestMessage::Tool(t) => !drop_ids.contains(&t.tool_call_id),
            _ => true,
        });
    }

    #[tokio::test]
    async fn rewriting_tool_may_consume_its_own_call() {
        let tool = Tool::new_rewriting(Arc::new(|_: Mark, mut messages| {
            async move {
                strip_named_tool_calls(&mut messages, "Mark");
                Ok(("consumed".to_owned(), messages))
            }
            .boxed()
        }))
        .unwrap();

        let replaced = Arc::new(std::sync::Mutex::new(false));
        let invocation = invocation_with_rewrite_tool(tool, replaced.clone());

        let sibling = dummy_tool_call_id("call_todo", "Todo", "{}");
        let mark = dummy_tool_call_id("call_mark", "Mark", "{}");
        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestUserMessageArgs::default()
                .content("go")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![sibling, mark.clone()])
                .build()
                .unwrap()
                .into(),
            tool_result_msg("call_todo", "todo-ok"),
        ];

        let content = invocation
            .execute_tool_serial(&mark, &mut messages)
            .await
            .unwrap();

        assert_eq!(content, "consumed");
        assert!(*replaced.lock().unwrap());
        assert!(!history_has_tool_call_id(&messages, "call_mark"));
        assert!(history_has_tool_call_id(&messages, "call_todo"));
        assert!(history_has_tool_result_id(&messages, "call_todo"));
        assert!(!history_has_tool_result_id(&messages, "call_mark"));
    }

    #[tokio::test]
    async fn rewriting_tool_consume_leaving_own_result_is_rejected() {
        let tool = Tool::new_rewriting(Arc::new(|_: Mark, mut messages| {
            async move {
                // Drop the call but leave a fabricated result — illegal consume.
                strip_named_tool_calls(&mut messages, "Mark");
                messages.push(tool_result_msg("call_mark", "leftover"));
                Ok(("consumed".to_owned(), messages))
            }
            .boxed()
        }))
        .unwrap();

        let replaced = Arc::new(std::sync::Mutex::new(false));
        let invocation = invocation_with_rewrite_tool(tool, replaced.clone());

        let mark = dummy_tool_call_id("call_mark", "Mark", "{}");
        let mut messages = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![mark.clone()])
                .build()
                .unwrap()
                .into(),
        ];

        let err = invocation
            .execute_tool_serial(&mark, &mut messages)
            .await
            .expect_err("consume must not leave its own tool result");
        assert!(
            err.to_string().contains("left a role=tool result"),
            "unexpected error: {err}"
        );
        assert!(!*replaced.lock().unwrap());
    }

    #[test]
    fn can_append_tool_result_accepts_assistant_or_sibling_tool_tail() {
        let mark = dummy_tool_call_id("c2", "Mark", "{}");
        let asst: ChatCompletionRequestMessage =
            ChatCompletionRequestAssistantMessageArgs::default()
                .tool_calls(vec![dummy_tool_call_id("c1", "Todo", "{}"), mark.clone()])
                .build()
                .unwrap()
                .into();
        let with_tail = vec![
            ChatCompletionRequestSystemMessageArgs::default()
                .content("sys")
                .build()
                .unwrap()
                .into(),
            asst.clone(),
            tool_result_msg("c1", "todo-ok"),
        ];
        assert!(can_append_tool_result(&with_tail, "c2"));
        assert!(!can_append_tool_result(&with_tail, "missing"));

        let asst_only = vec![asst];
        assert!(can_append_tool_result(&asst_only, "c2"));

        let user_last = vec![
            ChatCompletionRequestUserMessageArgs::default()
                .content("summary")
                .build()
                .unwrap()
                .into(),
        ];
        assert!(!can_append_tool_result(&user_last, "c2"));
        assert!(!can_append_tool_result(&[], "c2"));
    }

    #[tokio::test]
    async fn normal_tool_is_not_rewriting() {
        let tool = Tool::new(Arc::new(|_: NoArgs| {
            async move { Ok("ok".to_owned()) }.boxed()
        }))
        .unwrap();
        assert!(!tool.is_rewriting());
        assert_eq!(tool.name(), "NoArgs");
    }

    #[tokio::test]
    async fn run_without_client_requires_inference_callback() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        drop(tx);

        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .incoming(rx)
            .build()
            .unwrap();

        let err = invocation
            .run_without_client()
            .await
            .expect_err("should require inference_callback");
        assert!(
            err.to_string().contains("inference_callback"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn inference_callback_text_reply() {
        use std::sync::Mutex;

        let pushed = Arc::new(Mutex::new(Vec::<String>::new()));
        let pushed_cb = pushed.clone();
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);

        let (tx, rx) = tokio::sync::mpsc::channel(4);

        let cb: InferenceCallback = Arc::new(|messages| {
            async move {
                assert!(
                    messages.len() >= 2,
                    "expected system + user, got {}",
                    messages.len()
                );
                Ok(assistant_text("hello from mock"))
            }
            .boxed()
        });

        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .incoming(rx)
            .inference_callback(cb)
            .messages_push_callback({
                let done_tx = done_tx.clone();
                let cb: MessagesPushCallback = Arc::new(move |msg| {
                    let s = format!("{msg:?}");
                    if s.contains("hello from mock") {
                        let _ = done_tx.try_send(());
                    }
                    pushed_cb.lock().unwrap().push(s);
                });
                cb
            })
            .build()
            .unwrap();

        // Keep the sender open until the assistant turn is recorded; the loop
        // exits as soon as `incoming` is closed.
        let run = tokio::spawn(invocation.run_without_client());
        tx.send(UserRequest::Message(
            ChatCompletionRequestUserMessageContent::Text("hi".into()),
        ))
        .await
        .unwrap();
        done_rx.recv().await.expect("assistant reply");
        drop(tx);

        run.await.expect("join").unwrap();

        let log = pushed.lock().unwrap();
        assert!(
            log.iter().any(|s| s.contains("hello from mock")),
            "assistant text should be pushed; got {log:?}"
        );
    }

    #[tokio::test]
    async fn inference_callback_tool_then_text() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let tool_hits = Arc::new(AtomicUsize::new(0));
        let tool_hits_cb = tool_hits.clone();
        let tool = Tool::new(Arc::new(move |_: NoArgs| {
            let hits = tool_hits_cb.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                Ok("tool-ok".to_owned())
            }
            .boxed()
        }))
        .unwrap();

        let step = Arc::new(AtomicUsize::new(0));
        let saw_tool_result = Arc::new(AtomicUsize::new(0));
        let saw_tool_result_cb = saw_tool_result.clone();
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);

        let cb: InferenceCallback = Arc::new(move |messages| {
            let step = step.clone();
            let saw = saw_tool_result_cb.clone();
            let done_tx = done_tx.clone();
            async move {
                let n = step.fetch_add(1, Ordering::SeqCst);
                match n {
                    0 => {
                        assert!(
                            messages.len() >= 2,
                            "first turn: system + user, got {}",
                            messages.len()
                        );
                        Ok(assistant_tool_calls(vec![dummy_tool_call("NoArgs", "{}")]))
                    }
                    1 => {
                        // After tool execution, history should include a tool result.
                        let has_tool = messages
                            .iter()
                            .any(|m| matches!(m, ChatCompletionRequestMessage::Tool(_)));
                        if has_tool {
                            saw.fetch_add(1, Ordering::SeqCst);
                        }
                        let _ = done_tx.send(()).await;
                        Ok(assistant_text("all done"))
                    }
                    _ => Ok(assistant_text("extra")),
                }
            }
            .boxed()
        });

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .tools(vec![tool])
            .incoming(rx)
            .inference_callback(cb)
            .build()
            .unwrap();

        let run = tokio::spawn(invocation.run_without_client());
        tx.send(UserRequest::Message(
            ChatCompletionRequestUserMessageContent::Text("go".into()),
        ))
        .await
        .unwrap();
        done_rx.recv().await.expect("second inference turn");
        drop(tx);

        run.await.expect("join").unwrap();

        assert_eq!(tool_hits.load(Ordering::SeqCst), 1);
        assert_eq!(saw_tool_result.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn inference_identity_rewrite_mid_batch() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ping_hits = Arc::new(AtomicUsize::new(0));
        let ping_hits_cb = ping_hits.clone();
        let ping = Tool::new(Arc::new(move |_: NoArgs| {
            let hits = ping_hits_cb.clone();
            async move {
                let n = hits.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(format!("ping-{n}"))
            }
            .boxed()
        }))
        .unwrap();

        let mark = Tool::new_rewriting(Arc::new(|_: Mark, messages| {
            async move { Ok(("OK <fold-mark />".to_owned(), messages)) }.boxed()
        }))
        .unwrap();

        let replaced = Arc::new(AtomicUsize::new(0));
        let replaced_cb = replaced.clone();
        let step = Arc::new(AtomicUsize::new(0));
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);

        let cb: InferenceCallback = Arc::new(move |messages| {
            let step = step.clone();
            let done_tx = done_tx.clone();
            async move {
                let n = step.fetch_add(1, Ordering::SeqCst);
                match n {
                    0 => Ok(assistant_tool_calls(vec![
                        dummy_tool_call_id("c1", "NoArgs", "{}"),
                        dummy_tool_call_id("c2", "Mark", "{}"),
                        dummy_tool_call_id("c3", "NoArgs", "{}"),
                    ])),
                    1 => {
                        let texts: Vec<String> =
                            messages.iter().map(|m| format!("{m:?}")).collect();
                        let tool_n = texts
                            .iter()
                            .filter(|s| s.contains("tool_call_id") || s.contains("Tool"))
                            .count();
                        assert!(
                            texts.iter().any(|s| s.contains("ping-1")),
                            "first ping missing: {texts:?}"
                        );
                        assert!(
                            texts.iter().any(|s| s.contains("fold-mark")),
                            "identity mark result missing: {texts:?}"
                        );
                        assert!(
                            texts.iter().any(|s| s.contains("ping-2")),
                            "second ping missing: {texts:?}"
                        );
                        assert!(
                            tool_n >= 3,
                            "expected three tool results in history: {texts:?}"
                        );
                        let _ = done_tx.send(()).await;
                        Ok(assistant_text("done"))
                    }
                    _ => Ok(assistant_text("extra")),
                }
            }
            .boxed()
        });

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .tools(vec![ping, mark])
            .parallel_tools(true)
            .incoming(rx)
            .inference_callback(cb)
            .messages_replace_callback({
                let cb: MessagesReplaceCallback = Arc::new(move |_, _| {
                    replaced_cb.fetch_add(1, Ordering::SeqCst);
                });
                cb
            })
            .build()
            .unwrap();

        let run = tokio::spawn(invocation.run_without_client());
        tx.send(UserRequest::Message(
            ChatCompletionRequestUserMessageContent::Text("go".into()),
        ))
        .await
        .unwrap();
        done_rx.recv().await.expect("second inference turn");
        drop(tx);

        run.await.expect("join").unwrap();

        assert_eq!(ping_hits.load(Ordering::SeqCst), 2);
        assert_eq!(
            replaced.load(Ordering::SeqCst),
            0,
            "identity Mark must not fire replace callback"
        );
    }

    #[tokio::test]
    async fn inference_consume_own_call_skips_tool_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ping_hits = Arc::new(AtomicUsize::new(0));
        let ping_hits_cb = ping_hits.clone();
        let ping = Tool::new(Arc::new(move |_: NoArgs| {
            let hits = ping_hits_cb.clone();
            async move {
                let n = hits.fetch_add(1, Ordering::SeqCst) + 1;
                Ok(format!("ping-{n}"))
            }
            .boxed()
        }))
        .unwrap();

        let consume = Tool::new_rewriting(Arc::new(|_: Mark, mut messages| {
            async move {
                strip_named_tool_calls(&mut messages, "Mark");
                Ok(("SHOULD-NOT-APPEAR".to_owned(), messages))
            }
            .boxed()
        }))
        .unwrap();

        let step = Arc::new(AtomicUsize::new(0));
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);

        let cb: InferenceCallback = Arc::new(move |messages| {
            let step = step.clone();
            let done_tx = done_tx.clone();
            async move {
                let n = step.fetch_add(1, Ordering::SeqCst);
                match n {
                    0 => Ok(assistant_tool_calls(vec![
                        dummy_tool_call_id("c1", "NoArgs", "{}"),
                        dummy_tool_call_id("c2", "Mark", "{}"),
                        dummy_tool_call_id("c3", "NoArgs", "{}"),
                    ])),
                    1 => {
                        let texts: Vec<String> =
                            messages.iter().map(|m| format!("{m:?}")).collect();
                        assert!(
                            texts.iter().any(|s| s.contains("ping-1")),
                            "first ping missing: {texts:?}"
                        );
                        assert!(
                            texts.iter().any(|s| s.contains("ping-2")),
                            "second ping missing: {texts:?}"
                        );
                        assert!(
                            !texts.iter().any(|s| s.contains("SHOULD-NOT-APPEAR")),
                            "consumed call must not append a tool result: {texts:?}"
                        );
                        assert!(
                            !history_has_tool_call_id(&messages, "c2"),
                            "consumed Mark tool_call must be gone: {texts:?}"
                        );
                        let _ = done_tx.send(()).await;
                        Ok(assistant_text("done"))
                    }
                    _ => Ok(assistant_text("extra")),
                }
            }
            .boxed()
        });

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .tools(vec![ping, consume])
            .incoming(rx)
            .inference_callback(cb)
            .build()
            .unwrap();

        let run = tokio::spawn(invocation.run_without_client());
        tx.send(UserRequest::Message(
            ChatCompletionRequestUserMessageContent::Text("go".into()),
        ))
        .await
        .unwrap();
        done_rx.recv().await.expect("second inference turn");
        drop(tx);

        run.await.expect("join").unwrap();
        assert_eq!(ping_hits.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn inference_callback_error_surfaces() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);

        let cb: InferenceCallback =
            Arc::new(|_messages| async move { Err(anyhow::anyhow!("mock boom")) }.boxed());

        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .incoming(rx)
            .inference_callback(cb)
            .build()
            .unwrap();

        // Spawn first so the open sender is not closed before the loop runs.
        let run = tokio::spawn(invocation.run_without_client());
        tx.send(UserRequest::Message(
            ChatCompletionRequestUserMessageContent::Text("hi".into()),
        ))
        .await
        .unwrap();

        let err = run
            .await
            .expect("join")
            .expect_err("callback error should fail run");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("mock boom") || msg.contains("inference callback"),
            "unexpected error: {msg}"
        );
    }

    #[tokio::test]
    async fn seed_messages_appear_before_new_user_turn() {
        use std::sync::Mutex;

        let pushed = Arc::new(Mutex::new(Vec::<String>::new()));
        let pushed_cb = pushed.clone();
        let (done_tx, mut done_rx) = tokio::sync::mpsc::channel::<()>(1);

        let seed_user = ChatCompletionRequestUserMessageArgs::default()
            .content("prior user turn")
            .build()
            .unwrap()
            .into();
        let seed_asst = ChatCompletionRequestAssistantMessageArgs::default()
            .content("prior assistant turn")
            .build()
            .unwrap()
            .into();

        let cb: InferenceCallback = Arc::new(|messages| {
            async move {
                // system + 2 seed + new user
                assert!(
                    messages.len() >= 4,
                    "expected system + seed×2 + user, got {}",
                    messages.len()
                );
                let texts: Vec<String> = messages.iter().map(|m| format!("{m:?}")).collect();
                assert!(
                    texts.iter().any(|s| s.contains("prior user turn")),
                    "seed user missing: {texts:?}"
                );
                assert!(
                    texts.iter().any(|s| s.contains("prior assistant turn")),
                    "seed assistant missing: {texts:?}"
                );
                assert!(
                    texts.iter().any(|s| s.contains("fresh user")),
                    "new user missing: {texts:?}"
                );
                // seed comes before fresh user
                let seed_idx = texts
                    .iter()
                    .position(|s| s.contains("prior assistant turn"))
                    .unwrap();
                let fresh_idx = texts.iter().position(|s| s.contains("fresh user")).unwrap();
                assert!(seed_idx < fresh_idx);
                Ok(assistant_text("ok after seed"))
            }
            .boxed()
        });

        let (tx, rx) = tokio::sync::mpsc::channel(4);
        let invocation = AgentInvocationArgs::default()
            .system_prompt("sys")
            .incoming(rx)
            .seed_messages(vec![seed_user, seed_asst])
            .inference_callback(cb)
            .messages_push_callback({
                let done_tx = done_tx.clone();
                let cb: MessagesPushCallback = Arc::new(move |msg| {
                    let s = format!("{msg:?}");
                    if s.contains("ok after seed") {
                        let _ = done_tx.try_send(());
                    }
                    pushed_cb.lock().unwrap().push(s);
                });
                cb
            })
            .build()
            .unwrap();

        let run = tokio::spawn(invocation.run_without_client());
        // Seed must not alone trigger inference; send a fresh user message.
        tx.send(UserRequest::Message(
            ChatCompletionRequestUserMessageContent::Text("fresh user".into()),
        ))
        .await
        .unwrap();
        done_rx.recv().await.expect("assistant reply");
        drop(tx);

        run.await.expect("join").unwrap();

        let log = pushed.lock().unwrap();
        assert!(
            log.iter().any(|s| s.contains("prior user turn")),
            "seed user should be pushed; got {log:?}"
        );
        assert!(
            log.iter().any(|s| s.contains("prior assistant turn")),
            "seed assistant should be pushed; got {log:?}"
        );
    }
}
