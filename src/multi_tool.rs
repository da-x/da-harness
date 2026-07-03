use std::sync::Arc;

use async_openai::types::ChatCompletionTool;
use futures::future::BoxFuture;
use serde::Deserialize;

use crate::generate_tool_schema;

pub type TaskFuture = BoxFuture<'static, anyhow::Result<()>>;

pub struct Tool {
    handler: Arc<dyn Fn(serde_json::Value) -> TaskFuture + Send + Sync>,
    description: ChatCompletionTool,
}

impl Tool {
    fn new<T>(callback: Arc<dyn Fn(T) -> TaskFuture + Send + Sync>) -> anyhow::Result<Self>
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
}

impl AgentInvocation {
    async fn run() -> anyhow::Result<()> {
        todo!();
    }
}
