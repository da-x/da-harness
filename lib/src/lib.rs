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
use serde::{Deserialize, Serialize};
use tracing::warn;

// ─── OpenAIClient (taken as-is from shotef) ──────────────────────────

#[derive(Debug, Clone)]
pub struct LLMConfig {
    pub api_base: String,
    pub model_name: String,
    pub api_key: String,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_base: "http://127.0.0.1:4242/v1".to_owned(),
            model_name: "LocalModel".to_owned(),
            api_key: "".to_owned(),
        }
    }
}

#[derive(Clone)]
pub struct OpenAIClient {
    client: Client<OpenAIConfig>,
    model_name: String,
}

impl OpenAIClient {
    pub fn new() -> Self {
        Self::with_config(LLMConfig::default())
    }

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

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub async fn chat(&self, messages: Vec<ChatCompletionRequestMessage>, temperature: f32) -> anyhow::Result<String> {
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

// ─── LoopControl ─────────────────────────────────────────────────────

pub enum LoopControl<Agent: AgentLoop> {
    Continue(Agent::Request),
    Stop(Agent::Output),
}

// ─── AgentLoop Trait ─────────────────────────────────────────────────

#[async_trait]
pub trait AgentLoop: Sized {
    type Request: Serialize + Clone;
    type Response: for<'de> Deserialize<'de> + Serialize + Clone;
    type Output;

    fn system_prompt(&self) -> &str;
    fn user_prompt(&self) -> &str;
    fn examples(&self) -> Vec<(Self::Request, Self::Response)>;
    fn initial_input(&self) -> Self::Request;
    fn extend_prompt(&self, _history: &[(Self::Request, Self::Response)]) -> Option<String> {
        None
    }

    async fn iteration(
        &self,
        response: Self::Response,
        history: &[(Self::Request, Self::Response)],
    ) -> LoopControl<Self>;
}

// ─── run_loop ────────────────────────────────────────────────────────

pub async fn run_loop<Agent: AgentLoop>(
    client: OpenAIClient,
    agent: Agent,
) -> anyhow::Result<Agent::Output> {
    run_loop_with_max(client, agent, 10).await
}

pub async fn run_loop_with_max<Agent: AgentLoop>(
    client: OpenAIClient,
    agent: Agent,
    max_iterations: usize,
) -> anyhow::Result<Agent::Output> {
    let mut history: Vec<(Agent::Request, Agent::Response)> = Vec::new();
    let mut current_request = agent.initial_input();

    for _ in 0..max_iterations {
        let user_message = build_user_prompt(&agent, &history, &current_request);

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

        let response: Agent::Response =
            serde_json::from_str(&response_text).context(format!("failed to deserialize LLM response: {}", response_text))?;

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

fn build_user_prompt<Agent: AgentLoop>(
    agent: &Agent,
    history: &[(Agent::Request, Agent::Response)],
    current_request: &Agent::Request,
) -> String {
    let mut buf = String::new();

    buf.push_str(agent.user_prompt());
    buf.push('\n');

    let examples = agent.examples();
    if !examples.is_empty() {
        buf.push_str("## EXAMPLES\n\n");
        for (i, (req, resp)) in examples.iter().enumerate() {
            let req_json = serde_json::to_string(req).expect("serialize example request");
            let resp_json = serde_json::to_string(resp).expect("serialize example response");
            buf.push_str(&format!(
                "Example {}:\nINPUT (JSON): {}\nOUTPUT (JSON): {}\n\n",
                i + 1, req_json, resp_json
            ));
        }
    }

    if !history.is_empty() {
        buf.push_str("## CONVERSATION HISTORY\n\n");
        for (req, resp) in history {
            let req_json = serde_json::to_string(req).expect("serialize history request");
            let resp_json = serde_json::to_string(resp).expect("serialize history response");
            buf.push_str(&format!("INPUT (JSON): {}\nOUTPUT (JSON): {}\n\n", req_json, resp_json));
        }
    }

    if let Some(extended) = agent.extend_prompt(history) {
        buf.push_str("## ADDITIONAL CONTEXT\n");
        buf.push_str(&extended);
        buf.push('\n');
    }

    let current_json = serde_json::to_string(current_request).expect("serialize current request");
    buf.push_str("## CURRENT INPUT (JSON):\n");
    buf.push_str(&current_json);
    buf.push('\n');

    buf
}
