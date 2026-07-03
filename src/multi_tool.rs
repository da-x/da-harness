use std::sync::Arc;

use anyhow::Context;
use async_openai::types::{
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
    ChatCompletionRequestUserMessageArgs, ChatCompletionTool,
};
use futures::future::BoxFuture;
use serde::Deserialize;
use tracing::{debug, info};

use crate::{OpenAIClient, generate_tool_schema};

pub type TaskFuture = BoxFuture<'static, anyhow::Result<()>>;

#[derive(Clone)]
pub struct Tool {
    handler: Arc<dyn Fn(serde_json::Value) -> TaskFuture + Send + Sync>,
    description: ChatCompletionTool,
}

impl Tool {
    pub fn new<T>(callback: Arc<dyn Fn(T) -> TaskFuture + Send + Sync>) -> anyhow::Result<Self>
    where
        T: for<'de> Deserialize<'de> + Clone + schemars::JsonSchema + 'static,
    {
        let callback = callback;
        let handler: Arc<dyn Fn(serde_json::Value) -> TaskFuture + Send + Sync> =
            Arc::new(move |value: serde_json::Value| {
                let cb = callback.clone();
                Box::pin(async move {
                    let args: T = serde_json::from_value(value).expect("valid tool args");
                    cb(args).await
                })
            });

        let description = generate_tool_schema::<T>()?;

        Ok(Self {
            handler,
            description,
        })
    }
}

pub struct AgentInvocation {
    pub system_prompt: String,

    // Tools he agent can invoke. We use the tool's function name to differentiate
    // when the LLM tells us to execute something, and to invoke the correct handler
    // as an async tasks. If we receive a `tool_calls` vector of multple items we
    // issue the tasks in parallel and call `select!`, or call serially based o
    // `parallel_tools`.
    pub tools: Vec<Tool>,
    pub parallel_tools: bool,

    // Incoming user messages. Implementation will try_read from this between
    // LLM invocations. If closed, the loop ends.
    pub incoming: tokio::sync::mpsc::Receiver<String>,

    // Called when agent wants to say something
    pub agent_message_callback: Arc<dyn Fn(String) -> TaskFuture + Send + Sync>,

    // Called when agent wants to say something
    pub agent_idle_callback: Arc<dyn Fn() -> TaskFuture + Send + Sync>,
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

        // Track whether we have a pending user message to inject.
        loop {
            // If channel is closed and no pending work, we're done.
            if self.incoming.is_closed() {
                info!(target: "da_harness::multi_tool", "incoming channel closed, ending loop");
                break;
            }

            // Drain any available user messages from the incoming channel.
            let mut added = 0;
            while let Ok(msg) = self.incoming.try_recv() {
                debug!(target: "da_harness::multi_tool", msg = %msg, "<< user message received");
                messages.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(msg.as_str())
                        .build()
                        .context("building user message")?
                        .into(),
                );
                added += 1;
            }

            let (response, _prompt_tokens) = if prev_call || added > 0 {
                client
                    .chat_with_tools(messages.clone(), tools.clone(), self.parallel_tools, 0.6)
                    .await
                    .context("LLM chat call failed")?
            } else {
                (self.agent_idle_callback)().await?;

                if let Some(msg) = self.incoming.recv().await {
                    debug!(target: "da_harness::multi_tool", msg = %msg, "<< user message received");
                    messages.push(
                        ChatCompletionRequestUserMessageArgs::default()
                            .content(msg.as_str())
                            .build()
                            .context("building user message")?
                            .into(),
                    );
                    continue;
                } else {
                    break;
                }
            };

            if let Some(tool_calls) = &response.tool_calls {
                info!(
                    target: "da_harness::multi_tool",
                    n_tools = tool_calls.len(),
                    "<< LLM requested tool calls"
                );

                let mut asst_builder = ChatCompletionRequestAssistantMessageArgs::default();
                asst_builder.tool_calls(tool_calls.clone());
                if let Some(ref c) = response.content {
                    asst_builder.content(c.as_str());
                }
                let assistant_msg: ChatCompletionRequestMessage = asst_builder
                    .build()
                    .context("building assistant message")?
                    .into();

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
                    }
                }

                messages.push(assistant_msg);
                prev_call = true;
            } else {
                prev_call = false;
                let content = response.content.unwrap_or_default();
                info!(target: "da_harness::multi_tool", reply = %content, "<< LLM text response");

                let mut asst_builder = ChatCompletionRequestAssistantMessageArgs::default();
                if !content.is_empty() {
                    let _ = (self.agent_message_callback)(content.clone()).await;
                    asst_builder.content(content.as_str());
                }

                messages.push(
                    asst_builder
                        .build()
                        .context("building assistant message")?
                        .into(),
                );
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

        Box::pin(async move {
            let tool = tools
                .iter()
                .find(|t| t.description.function.name.as_str() == name)
                .ok_or_else(|| anyhow::anyhow!("unknown tool: {}", name))?;

            let parsed: serde_json::Value =
                serde_json::from_str(&args).context("failed to parse tool arguments")?;
            (tool.handler)(parsed).await?;
            Ok("OK".to_string())
        })
    }
}
