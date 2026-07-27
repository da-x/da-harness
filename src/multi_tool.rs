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
//! automatic token-threshold trigger in this module (`prompt_tokens` from each
//! chat call are currently unused — hosts may surface usage in the system
//! prompt or pass explicit tool args such as `keep_last`).
//!
//! ## Protocol invariants for rewriting handlers
//!
//! When the handler runs, `messages` already includes the assistant message
//! that requested this tool call (and any earlier tool results from the same
//! serial batch). The returned list **must**:
//!
//! 1. Be a legal chat prefix for appending one more `role=tool` message with
//!    this call's `tool_call_id`.
//! 2. Typically end with that same assistant `tool_calls` message (or an
//!    equivalent one that still lists this id).
//! 3. **Not** already include the tool response for this call — the loop
//!    appends it after the handler returns.
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
//! sees only tool results already appended earlier in the same batch. Prefer
//! calling rewrite tools alone (or last) from the model side.
//!
//! ## Persistence
//!
//! - [`AgentInvocation::messages_push_callback`] — append-only mirror of each
//!   new message.
//! - [`AgentInvocation::messages_replace_callback`] — full replace after a
//!   rewriting tool, with `(old_messages, new_messages)`; hosts that persist
//!   history should treat the new list as authoritative (do not rely on push
//!   alone after a rewrite).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
    ChatCompletionTool,
};
use futures::{FutureExt, future::BoxFuture};
use serde::Deserialize;
use tokio_retry::Retry;
use tracing::{debug, info, warn};

use crate::{OpenAIClient, generate_tool_schema};

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
    /// `new_messages`, fires [`AgentInvocation::messages_replace_callback`], then
    /// appends a normal `role=tool` message with `tool_content`.
    ///
    /// Handlers that need async work (e.g. an LLM summarization call) can freely
    /// `.await` because they own the message snapshot — see module docs for why
    /// this is not `&mut Vec`.
    ///
    /// # Protocol
    ///
    /// `new_messages` must remain a valid prefix for appending this tool's
    /// response (usually keep the trailing assistant `tool_calls` message).
    /// See the [module-level docs](self).
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

    /// Optional [tokio-retry](https://crates.io/crates/tokio-retry) strategy for each
    /// `chat_with_tools` LLM call.
    ///
    /// When set, a fresh strategy is produced for every request and transient failures
    /// are retried with the strategy's delays between attempts. Defaults to no retries
    /// (a single attempt).
    ///
    /// Use [`AgentInvocationArgs::retry_strategy`] to set this from any cloneable
    /// `IntoIterator<Item = Duration>` (e.g. `ExponentialBackoff`, `FixedInterval`).
    #[builder(default, setter(custom))]
    pub retry_strategy: Option<RetryStrategyFactory>,
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

impl AgentInvocation {
    pub async fn run(mut self, client: OpenAIClient) -> anyhow::Result<()> {
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

            let (response, _prompt_tokens) = if prev_call || new_user_messages > 0 {
                new_user_messages = 0;
                // `_prompt_tokens` is available for hosts that want usage-aware
                // prompts; multi_tool does not auto-compact on thresholds.
                self.chat_with_tools_retrying(&client, messages.clone(), tools.clone(), temperature)
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
                    // After each rewrite, `messages` is the handler's replacement list;
                    // we then append this call's tool result (protocol requires it).
                    for tc in tool_calls {
                        let content = match self.execute_tool_serial(tc, &mut messages).await {
                            Ok(s) => s,
                            Err(e) => format!("Error: {}", e),
                        };
                        self.push_tool_result(&mut messages, tc, content)?;
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
    ///   Does **not** append the tool result; the caller does that next.
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
                // Soft check: last message should still be the assistant tool_calls
                // turn so appending this tool response stays protocol-valid.
                anyhow::ensure!(
                    matches!(
                        new_messages.last(),
                        Some(ChatCompletionRequestMessage::Assistant(_))
                    ),
                    "rewriting tool '{}' should leave a trailing assistant message \
                     (usually the tool_calls turn) so the tool result can be appended",
                    name
                );

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

    /// Call `chat_with_tools`, optionally retrying with the configured strategy.
    async fn chat_with_tools_retrying(
        &self,
        client: &OpenAIClient,
        messages: Vec<ChatCompletionRequestMessage>,
        tools: Vec<ChatCompletionTool>,
        temperature: f32,
    ) -> anyhow::Result<(
        async_openai::types::ChatCompletionResponseMessage,
        Option<u32>,
    )> {
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
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        ChatCompletionToolType, FunctionCall,
    };

    /// Minimal tool schema args for unit tests.
    #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
    struct NoArgs {}

    #[derive(Debug, Clone, Deserialize, schemars::JsonSchema)]
    struct KeepLast {
        /// Number of trailing messages to keep after the system message.
        n: usize,
    }

    fn dummy_tool_call(name: &str, args: &str) -> ChatCompletionMessageToolCall {
        ChatCompletionMessageToolCall {
            id: "call_test".into(),
            r#type: ChatCompletionToolType::Function,
            function: FunctionCall {
                name: name.into(),
                arguments: args.into(),
            },
        }
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
                .content("asst")
                .build()
                .unwrap()
                .into(),
        ];
        // Pretend the last message is the tool_calls turn; we only check rewrite length.
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

    #[tokio::test]
    async fn normal_tool_is_not_rewriting() {
        let tool = Tool::new(Arc::new(|_: NoArgs| {
            async move { Ok("ok".to_owned()) }.boxed()
        }))
        .unwrap();
        assert!(!tool.is_rewriting());
        assert_eq!(tool.name(), "NoArgs");
    }
}
