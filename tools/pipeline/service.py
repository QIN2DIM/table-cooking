import io
from typing import Any
from urllib.parse import urlparse

import httpx
import pandas as pd
import tiktoken
from dify_plugin.core.runtime import Session
from dify_plugin.entities.model.llm import LLMModelConfig
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from dify_plugin.file.file import File, DIFY_FILE_IDENTITY, FileType
from loguru import logger
from pydantic import BaseModel, Field
import filetype

from tools.ai.table_self_query import TableQueryEngine, QueryResult

WRAPPER_HUMAN_READY = """
### Query code

```python
{py_code}
```

### Preview of execution results

{result_markdown}
"""

WRAPPER_LLM_READY = """
```xml
<segment table="{table_name}">
<question>{question}</question>
<code>
{query_code}
</code>
<output filename="{recommend_filename}">
{result_markdown}
</output>
</segment>
```
"""

encoding = tiktoken.get_encoding("o200k_base")


class ArtifactPayload(BaseModel):
    natural_query: str = Field(..., description="自然语言查询描述")
    table: File = Field(..., description="Dify表格文件")
    dify_model_config: LLMModelConfig = Field(..., description="Dify LLM模型配置")
    enable_classifier: bool = Field(
        default=True, description="启用问题分类器，将查询流引导至'简单查询'或'复杂计算'"
    )

    def get_table_stream(self) -> io.BytesIO:
        """获取表格文件的字节流"""
        return io.BytesIO(self.table.blob)

    @staticmethod
    def _validate_query(query: Any) -> None:
        """验证查询参数"""
        if not query:
            raise ToolProviderCredentialValidationError("查询参数不能为空")
        if not isinstance(query, str):
            raise ToolProviderCredentialValidationError("查询参数必须是字符串类型")
        if len(query.strip()) < 3:
            raise ToolProviderCredentialValidationError("查询参数过短，请提供更具体的查询描述")
        if len(query) > 1000:
            raise ToolProviderCredentialValidationError("查询参数过长，请限制在1000字符以内")

    @staticmethod
    def _validate_url(url: Any) -> None:
        """验证URL格式"""
        if not url:
            raise ToolProviderCredentialValidationError("URL不能为空")
        if not isinstance(url, str):
            raise ToolProviderCredentialValidationError("URL必须是字符串类型")

        try:
            parsed_url = urlparse(url)

            # 验证URL方案
            if parsed_url.scheme not in ["http", "https"]:
                scheme = parsed_url.scheme or "缺失"
                raise ToolProviderCredentialValidationError(
                    f"无效的URL方案 '{scheme}'。表格文件链接必须以 http:// 或 https:// 开头"
                )

            # 验证URL路径
            if not parsed_url.path or parsed_url.path == "/":
                raise ToolProviderCredentialValidationError("URL缺少有效的文件路径")

            # 验证URL主机名
            if not parsed_url.netloc:
                raise ToolProviderCredentialValidationError("URL缺少有效的主机名")

            # 验证URL长度
            if len(url) > 2048:
                raise ToolProviderCredentialValidationError("URL长度超过限制，请提供更短的URL")
        except Exception as e:
            if isinstance(e, ToolProviderCredentialValidationError):
                raise
            raise ToolProviderCredentialValidationError(f"URL解析错误: {str(e)}")

    @staticmethod
    def _validate_file_extension(extension: str) -> None:
        """验证文件扩展名"""
        valid_extensions = [".csv", ".xls", ".xlsx"]
        if not extension or extension not in valid_extensions:
            raise ToolProviderCredentialValidationError(
                f"不支持的文件类型：{extension or '未知'}。仅支持以下格式：{', '.join(valid_extensions)}"
            )

    @staticmethod
    def _validate_file_size(size: int) -> None:
        """验证文件大小"""
        max_file_size = 1000 * 1024 * 1024  # 1000MB
        if size > max_file_size:
            raise ToolProviderCredentialValidationError(
                f"文件大小超过限制，最大允许1000MB，当前大小：{size // (1024 * 1024)}MB"
            )

    @staticmethod
    def _validate_model_availability(chef: dict) -> None:
        """验证模型可用性，防止使用不恰当的高成本模型"""
        not_available_models = [
            "gpt-4.5-preview",
            "gpt-4.5-preview-2025-02-27",
            "o1",
            "o1-2024-12-17",
            "o1-pro",
            "o1-pro-2025-03-19",
        ]
        if (
            isinstance(chef, dict)
            and chef.get("model_type", "") == "llm"
            and chef.get("provider", "") == "langgenius/openai/openai"
            and chef.get("mode", "") == "chat"
        ):
            if use_model := chef.get("model"):
                if use_model in not_available_models:
                    raise ToolProviderCredentialValidationError(
                        f"模型 `{use_model}` 不可用于此工具。请替换为其他更经济的模型。"
                    )

    @staticmethod
    def _validate_table_file(table: Any) -> None:
        """验证表格文件对象"""
        if not table:
            raise ToolProviderCredentialValidationError("表格参数不能为空")
        if not isinstance(table, File):
            raise ToolProviderCredentialValidationError("表格参数必须是File类型对象")

        # 验证文件URL
        if not hasattr(table, "url") or not table.url:
            raise ToolProviderCredentialValidationError("表格URL不能为空")

        ArtifactPayload._validate_url(table.url)
        ArtifactPayload._validate_file_extension(table.extension)
        ArtifactPayload._validate_file_size(table.size)

    @staticmethod
    def _validate_chef(chef: Any) -> None:
        """验证chef参数"""
        if not chef:
            raise ToolProviderCredentialValidationError("chef参数不能为空")
        if not isinstance(chef, dict):
            raise ToolProviderCredentialValidationError("chef参数必须是模型对象(Object)")

        ArtifactPayload._validate_model_availability(chef)

    @staticmethod
    def fetch_table(file_url: str) -> File:
        """从URL获取表格文件，不持久化

        Args:
            file_url: 文件URL

        Returns:
            File对象
        """
        # 解析URL以获取文件信息
        file_url = file_url.strip()
        parsed_url = urlparse(file_url)
        path_parts = parsed_url.path.split("/")
        filename = path_parts[-1] if path_parts else "unknown_file"

        # 获取文件内容
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        }
        try:
            response = httpx.get(file_url, headers=headers)
            response.raise_for_status()
            _blob = response.content
        except Exception as e:
            raise ToolProviderCredentialValidationError(f"获取文件失败: {str(e)}")

        # 提取扩展名
        extension = ""
        kind = filetype.guess(_blob)
        if kind is not None:
            extension = f".{kind.extension}"
        elif "." in filename:
            # 回退到文件名判断
            extension = f".{filename.split('.')[-1].lower()}"

        # 验证文件类型
        ArtifactPayload._validate_file_extension(extension)

        # 获取MIME类型和文件大小
        mime_type = response.headers.get("content-type", None)
        size = len(_blob)

        # 验证文件大小
        ArtifactPayload._validate_file_size(size)

        return File(
            dify_model_identity=DIFY_FILE_IDENTITY,
            url=file_url,
            mime_type=mime_type,
            filename=filename,
            extension=extension,
            size=size,
            type=FileType.DOCUMENT,
            _blob=_blob,
        )

    @classmethod
    def from_dify(
        cls, tool_parameters: dict[str, Any], *, enable_classifier: bool = True
    ) -> "ArtifactPayload":
        """从Dify工具参数创建ArtifactPayload实例

        Args:
            tool_parameters: Dify工具参数
            enable_classifier: 是否启用分类器

        Returns:
            ArtifactPayload实例
        """
        if not tool_parameters or not isinstance(tool_parameters, dict):
            raise ToolProviderCredentialValidationError("工具参数必须是有效的字典类型")

        # 提取参数
        query = tool_parameters.get("query")
        table = tool_parameters.get("table")
        chef = tool_parameters.get("chef")

        # 验证参数
        cls._validate_query(query)
        cls._validate_table_file(table)
        cls._validate_chef(chef)

        return cls(
            natural_query=query,
            table=table,
            dify_model_config=chef,
            enable_classifier=enable_classifier,
        )

    @classmethod
    def from_s3(
        cls, tool_parameters: dict[str, Any], *, enable_classifier: bool = True
    ) -> "ArtifactPayload":
        """从S3参数创建ArtifactPayload实例

        Args:
            tool_parameters: 包含查询、文件URL和chef的参数
            enable_classifier: 是否启用分类器

        Returns:
            ArtifactPayload实例
        """
        if not tool_parameters or not isinstance(tool_parameters, dict):
            raise ToolProviderCredentialValidationError("工具参数必须是有效的字典类型")

        # 提取参数
        query = tool_parameters.get("query")
        file_url = tool_parameters.get("file_url")
        chef = tool_parameters.get("chef")

        # 验证参数
        cls._validate_query(query)
        cls._validate_url(file_url)
        cls._validate_chef(chef)

        # 获取表格文件
        table = cls.fetch_table(file_url)

        return cls(
            natural_query=query,
            table=table,
            dify_model_config=chef,
            enable_classifier=enable_classifier,
        )


class CodeInterpreter(BaseModel):
    code: str


class CookingResultParams(BaseModel):
    code: str
    natural_query: str
    recommend_filename: str
    input_tokens: int
    input_table_name: str


class CookingResult(BaseModel):
    llm_ready: str
    human_ready: str
    params: CookingResultParams


def transform_friendly_prompt_template(
    question: str, table_name: str, query_code: str, recommend_filename: str, result_data: Any
):
    preview_df = pd.DataFrame.from_records(result_data)
    result_markdown = preview_df.to_markdown(index=False)

    human_ready = WRAPPER_HUMAN_READY.format(py_code=query_code, result_markdown=result_markdown)

    llm_ready = WRAPPER_LLM_READY.format(
        question=question,
        table_name=table_name,
        query_code=query_code,
        recommend_filename=recommend_filename,
        result_markdown=result_markdown,
    ).strip()

    return llm_ready, human_ready


@logger.catch
def table_self_query(artifact: ArtifactPayload, session: Session) -> CookingResult | None:
    engine = TableQueryEngine(session=session, dify_model_config=artifact.dify_model_config)
    engine.load_table(file_stream=artifact.get_table_stream(), extension=artifact.table.extension)

    result: QueryResult = engine.query(artifact.natural_query)
    if not result:
        return

    if result.error:
        logger.error(result.error)

    recommend_filename = result.get_recommend_filename(suffix=".md")

    # ====================================================
    # Convert answer to XML format content of LLM_READY
    # ====================================================
    # Since the query result data volume may be very large,
    # it is not advisable to insert the complete content into the session polluting context.
    # The best practice is to insert preview lines and resource preview links
    __xml_context__, __preview_context__ = transform_friendly_prompt_template(
        question=artifact.natural_query,
        table_name=artifact.table.filename,
        query_code=result.query_code,
        recommend_filename=recommend_filename,
        result_data=result.data,
    )

    # ==========================================================================
    # Excessively long text should be printed directly instead of output by LLM
    # ==========================================================================
    input_tokens = len(encoding.encode(__xml_context__))

    # ==========================================================================
    # todo: Return to the table preview file after operation
    # ==========================================================================

    return CookingResult(
        llm_ready=__xml_context__,
        human_ready=__preview_context__,
        params=CookingResultParams(
            code=result.query_code,
            natural_query=artifact.natural_query,
            recommend_filename=recommend_filename,
            input_tokens=input_tokens,
            input_table_name=artifact.table.filename,
        ),
    )
