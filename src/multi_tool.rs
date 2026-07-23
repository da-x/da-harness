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

fn default_push_callback() -> Arc<dyn Fn(&ChatCompletionRequestMessage) + Send + Sync> {
    Arc::new(|_: &ChatCompletionRequestMessage| {})
}

#[derive(Clone)]
pub struct Tool {
    handler: Arc<dyn Fn(serde_json::Value) -> TaskFutureStr + Send + Sync>,
    description: ChatCompletionTool,
}

impl Tool {
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
            handler,
            description,
        })
    }

    /// OpenAI function name for this tool (from the generated schema).
    pub fn name(&self) -> &str {
        self.description.function.name.as_str()
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
    /// issue the tasks in parallel or serially based on `parallel_tools`.
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
    pub messages_push_callback: Arc<dyn Fn(&ChatCompletionRequestMessage) + Send + Sync>,

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
                self.chat_with_tools_retrying(
                    &client,
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

                if self.parallel_tools {
                    let futures: Vec<_> =
                        tool_calls.iter().map(|tc| self.execute_tool(tc)).collect();
                    let results = futures::future::join_all(futures).await;

                    for (tc, result) in tool_calls.iter().zip(results) {
                        let content = match result {
                            Ok(s) => s,
                            Err(e) => format!("Error: {}", e),
                        };
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
                    }
                } else {
                    for tc in tool_calls {
                        let content = match self.execute_tool(tc).await {
                            Ok(s) => s,
                            Err(e) => format!("Error: {}", e),
                        };
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
                    }
                }
            }
        }

        Ok(())
    }

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
            let res = (tool.handler)(parsed).await?;
            Ok(res)
        }
        .boxed()
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
