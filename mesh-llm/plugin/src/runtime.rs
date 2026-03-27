use anyhow::{Result, bail};
use rmcp::model::{
    CallToolResult, CancelTaskParams, CancelTaskResult, CompleteRequestParams, CompleteResult,
    GetPromptRequestParams, GetPromptResult, GetTaskInfoParams, GetTaskPayloadResult,
    GetTaskResult, GetTaskResultParams, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListTasksResult, ListToolsResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, ServerInfo, SetLevelRequestParams,
    SubscribeRequestParams, UnsubscribeRequestParams,
};

use crate::{
    context::PluginContext,
    error::{PluginError, PluginResult, PluginRpcResult},
        helpers::{
        ToolCallRequest, json_response, parse_get_prompt_request, parse_read_resource_request,
        parse_rpc_params, parse_tool_call_request,
    },
    io::{LocalStream, connect_from_env, read_envelope, write_envelope},
    proto,
    PROTOCOL_VERSION,
};

#[crate::async_trait]
pub trait Plugin: Send {
    fn plugin_id(&self) -> &str;
    fn plugin_version(&self) -> String;
    fn server_info(&self) -> ServerInfo;

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    async fn on_initialized(&mut self, _context: &mut PluginContext<'_>) -> Result<()> {
        Ok(())
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        Ok("ok".into())
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(None)
    }

    async fn call_tool(
        &mut self,
        _request: ToolCallRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        Ok(None)
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(None)
    }

    async fn get_prompt(
        &mut self,
        _request: GetPromptRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        Ok(None)
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(None)
    }

    async fn read_resource(
        &mut self,
        _request: ReadResourceRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        Ok(None)
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(None)
    }

    async fn subscribe_resource(
        &mut self,
        _request: SubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn unsubscribe_resource(
        &mut self,
        _request: UnsubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn complete(
        &mut self,
        _request: CompleteRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        Ok(None)
    }

    async fn set_log_level(
        &mut self,
        _request: SetLevelRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn list_tasks(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        Ok(None)
    }

    async fn get_task_info(
        &mut self,
        _request: GetTaskInfoParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        Ok(None)
    }

    async fn get_task_result(
        &mut self,
        _request: GetTaskResultParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        Ok(None)
    }

    async fn cancel_task(
        &mut self,
        _request: CancelTaskParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        Ok(None)
    }

    async fn handle_rpc(
        &mut self,
        request: proto::RpcRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        match request.method.as_str() {
            "tools/list" => match self.list_tools(context).await? {
                Some(result) => json_response(&result),
                None => Err(PluginError::method_not_found(
                    "Unsupported MCP method 'tools/list'",
                )),
            },
            "tools/call" => {
                let tool_call = parse_tool_call_request(&request)?;
                match self.call_tool(tool_call, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tools/call'",
                    )),
                }
            }
            "prompts/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_prompts(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/list'",
                    )),
                }
            }
            "prompts/get" => {
                let params = parse_get_prompt_request(&request)?;
                match self.get_prompt(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/get'",
                    )),
                }
            }
            "resources/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resources(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/list'",
                    )),
                }
            }
            "resources/read" => {
                let params = parse_read_resource_request(&request)?;
                match self.read_resource(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/read'",
                    )),
                }
            }
            "resources/templates/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resource_templates(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/templates/list'",
                    )),
                }
            }
            "resources/subscribe" => {
                let params: SubscribeRequestParams = parse_rpc_params(&request)?;
                match self.subscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/subscribe'",
                    )),
                }
            }
            "resources/unsubscribe" => {
                let params: UnsubscribeRequestParams = parse_rpc_params(&request)?;
                match self.unsubscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/unsubscribe'",
                    )),
                }
            }
            "completion/complete" => {
                let params: CompleteRequestParams = parse_rpc_params(&request)?;
                match self.complete(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'completion/complete'",
                    )),
                }
            }
            "logging/setLevel" => {
                let params: SetLevelRequestParams = parse_rpc_params(&request)?;
                match self.set_log_level(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'logging/setLevel'",
                    )),
                }
            }
            "tasks/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_tasks(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/list'",
                    )),
                }
            }
            "tasks/get" => {
                let params: GetTaskInfoParams = parse_rpc_params(&request)?;
                match self.get_task_info(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/get'",
                    )),
                }
            }
            "tasks/result" => {
                let params: GetTaskResultParams = parse_rpc_params(&request)?;
                match self.get_task_result(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/result'",
                    )),
                }
            }
            "tasks/cancel" => {
                let params: CancelTaskParams = parse_rpc_params(&request)?;
                match self.cancel_task(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/cancel'",
                    )),
                }
            }
            _ => Err(PluginError::method_not_found(format!(
                "Unsupported MCP method '{}'",
                request.method
            ))),
        }
    }

    async fn on_rpc_notification(
        &mut self,
        _notification: proto::RpcNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_channel_message(
        &mut self,
        _message: proto::ChannelMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_bulk_transfer_message(
        &mut self,
        _message: proto::BulkTransferMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_mesh_event(
        &mut self,
        _event: proto::MeshEvent,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_host_error(
        &mut self,
        error: proto::ErrorResponse,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        bail!("host error: {}", error.message)
    }
}

pub struct PluginRuntime;

impl PluginRuntime {
    pub async fn run<P: Plugin>(plugin: P) -> Result<()> {
        let stream = connect_from_env().await?;
        Self::run_with_stream(plugin, stream).await
    }

    pub async fn run_with_stream<P: Plugin>(mut plugin: P, mut stream: LocalStream) -> Result<()> {
        loop {
            let envelope = read_envelope(&mut stream).await?;
            let request_id = envelope.request_id;
            let plugin_id = plugin.plugin_id().to_string();

            match envelope.payload {
                Some(proto::envelope::Payload::InitializeRequest(_)) => {
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::InitializeResponse(
                            proto::InitializeResponse {
                                plugin_id: plugin_id.clone(),
                                plugin_protocol_version: PROTOCOL_VERSION,
                                plugin_version: plugin.plugin_version(),
                                server_info_json: serde_json::to_string(&plugin.server_info())?,
                                capabilities: plugin.capabilities(),
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_initialized(&mut context).await?;
                }
                Some(proto::envelope::Payload::HealthRequest(_)) => {
                    let detail = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        plugin.health(&mut context).await?
                    };
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::HealthResponse(
                            proto::HealthResponse {
                                status: proto::health_response::Status::Ok as i32,
                                detail,
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                }
                Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::ShutdownResponse(
                            proto::ShutdownResponse {},
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    break;
                }
                Some(proto::envelope::Payload::RpcRequest(request)) => {
                    let payload = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        match plugin.handle_rpc(request, &mut context).await {
                            Ok(payload) => payload,
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(err.into_error_response())
                            }
                        }
                    };
                    write_envelope(
                        &mut stream,
                        &proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(payload),
                        },
                    )
                    .await?;
                }
                Some(proto::envelope::Payload::RpcNotification(notification)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_rpc_notification(notification, &mut context).await?;
                }
                Some(proto::envelope::Payload::ChannelMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_channel_message(message, &mut context).await?;
                }
                Some(proto::envelope::Payload::BulkTransferMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_bulk_transfer_message(message, &mut context).await?;
                }
                Some(proto::envelope::Payload::MeshEvent(event)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_mesh_event(event, &mut context).await?;
                }
                Some(proto::envelope::Payload::ErrorResponse(error)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_host_error(error, &mut context).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}
