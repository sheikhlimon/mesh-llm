use anyhow::Result;
use serde::Serialize;

use crate::{
    PROTOCOL_VERSION,
    io::{LocalStream, send_bulk_transfer_message, send_channel_message, write_envelope},
    proto,
};

pub struct PluginContext<'a> {
    pub(crate) stream: &'a mut LocalStream,
    pub(crate) plugin_id: &'a str,
}

impl<'a> PluginContext<'a> {
    pub async fn send_channel(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.send_channel_message(message).await
    }

    pub async fn send_channel_message(&mut self, message: proto::ChannelMessage) -> Result<()> {
        send_channel_message(self.stream, self.plugin_id, message).await
    }

    pub async fn send_bulk(&mut self, message: proto::BulkTransferMessage) -> Result<()> {
        self.send_bulk_transfer_message(message).await
    }

    pub async fn send_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        send_bulk_transfer_message(self.stream, self.plugin_id, message).await
    }

    pub async fn notify_host<P>(&mut self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        write_envelope(
            self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.plugin_id.to_string(),
                request_id: 0,
                payload: Some(proto::envelope::Payload::RpcNotification(
                    proto::RpcNotification {
                        method: method.to_string(),
                        params_json: serde_json::to_string(&params)?,
                    },
                )),
            },
        )
        .await
    }
}
