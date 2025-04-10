from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from loguru import logger


class TableCookingS3Tool(Tool):
    """
    Invoke model:
    https://docs.dify.ai/zh-hans/plugins/schema-definition/reverse-invocation-of-the-dify-service/model#zui-jia-shi-jian
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        if tables := tool_parameters.get("tables"):
            for table in tables:
                logger.debug(f"获取文件链接 - {table.filename=}")
                yield self.create_text_message(table.url)
                yield self.create_json_message(table.model_dump(mode="json"))
        yield self.create_text_message("")
