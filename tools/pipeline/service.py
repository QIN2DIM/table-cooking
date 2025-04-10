import io
from typing import Any, NoReturn
from urllib.parse import urlparse

import httpx
import pandas as pd
import tiktoken
from dify_plugin.core.runtime import Session
from dify_plugin.entities.model.llm import LLMModelConfig
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from dify_plugin.file.file import File, DIFY_FILE_IDENTITY, FileType
from loguru import logger
from pydantic import BaseModel

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
    natural_query: str
    """
    Natural language query description.
    Todo: In multiple rounds of dialogue, this should be a semantic complete query after being spliced by the memory model.
    """

    table: File
    """
    Dify table
    """

    dify_model_config: LLMModelConfig
    """
    Dify LLM model configuration
    """

    enable_classifier: bool = True
    """
    Start the problem classifier and let the query flow to `simple query` or `complex calculation`
    """

    def get_table_stream(self) -> io.BytesIO:
        return io.BytesIO(self.table.blob)

    @staticmethod
    def validation_prevent_stupidity(chef: dict):
        # Prevent stupidity
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
                        f"Model `{use_model}` is not available for this tool. "
                        f"Please replace other cheaper models."
                    )

    @staticmethod
    def validation(tool_parameters: dict[str, Any]) -> NoReturn | None:
        # 首先验证tool_parameters本身
        if not tool_parameters or not isinstance(tool_parameters, dict):
            raise ToolProviderCredentialValidationError("工具参数必须是有效的字典类型")

        # 提取所需的关键参数
        query = tool_parameters.get("query")
        table = tool_parameters.get("table")
        chef = tool_parameters.get("chef")

        # !!<LLM edit>
        # 1. 查询参数验证
        if not query:
            raise ToolProviderCredentialValidationError("查询参数不能为空")
        if not isinstance(query, str):
            raise ToolProviderCredentialValidationError("查询参数必须是字符串类型")
        if len(query.strip()) < 3:
            raise ToolProviderCredentialValidationError("查询参数过短，请提供更具体的查询描述")
        if len(query) > 1000:
            raise ToolProviderCredentialValidationError("查询参数过长，请限制在1000字符以内")

        # 2. 表格文件验证
        if not table:
            raise ToolProviderCredentialValidationError("表格参数不能为空")
        if not isinstance(table, File):
            raise ToolProviderCredentialValidationError("表格参数必须是File类型对象")

        # 验证文件扩展名
        valid_extensions = [".csv", ".xls", ".xlsx"]
        if not table.extension or table.extension not in valid_extensions:
            raise ToolProviderCredentialValidationError(
                f"不支持的文件类型：{table.extension or '未知'}。仅支持以下格式：{', '.join(valid_extensions)}"
            )

        # 验证文件大小
        max_file_size = 1000 * 1024 * 1024  # 1000MB
        if table.size > max_file_size:
            raise ToolProviderCredentialValidationError(
                f"文件大小超过限制，最大允许1000MB，当前大小：{table.size // (1024 * 1024)}MB"
            )

        # 3. URL验证
        if not hasattr(table, "url") or not table.url:
            raise ToolProviderCredentialValidationError("表格URL不能为空")
        if not isinstance(table.url, str):
            raise ToolProviderCredentialValidationError("表格URL必须是字符串类型")

        # 解析并验证URL
        try:
            parsed_url = urlparse(table.url)

            # 验证URL方案
            if parsed_url.scheme not in ["http", "https"]:
                scheme = parsed_url.scheme or "缺失"
                raise ToolProviderCredentialValidationError(
                    f"无效的URL方案 '{scheme}'。表格文件链接必须以 http:// 或 https:// 开头\n"
                    f"你需要修改 .env 中的 FILES_URL 环境变量 \n"
                    f">>> https://github.com/langgenius/dify/blob/72191f5b13c55b44bcd3b25f7480804259e53495/docker/.env.example#L42"
                )

            # 验证URL路径
            if not parsed_url.path or parsed_url.path == "/":
                raise ToolProviderCredentialValidationError("URL缺少有效的文件路径")

            # 验证URL主机名
            if not parsed_url.netloc:
                raise ToolProviderCredentialValidationError("URL缺少有效的主机名")

            # 验证URL长度
            if len(table.url) > 2048:
                raise ToolProviderCredentialValidationError("URL长度超过限制，请提供更短的URL")
        except Exception as e:
            if isinstance(e, ToolProviderCredentialValidationError):
                raise
            raise ToolProviderCredentialValidationError(f"URL解析错误: {str(e)}")

        # !!</LLM edit>

        # 验证模型可用性
        ArtifactPayload.validation_prevent_stupidity(chef)

    @classmethod
    def from_dify(cls, tool_parameters: dict[str, Any], *, enable_classifier: bool = True):
        ArtifactPayload.validation(tool_parameters)
        return cls(
            natural_query=tool_parameters.get("query"),
            dify_model_config=tool_parameters.get("chef"),
            table=tool_parameters.get("table"),
            enable_classifier=enable_classifier,
        )

    @staticmethod
    def validation_s3(tool_parameters: dict[str, Any]) -> NoReturn | None:
        query = tool_parameters.get("query")
        file_url = tool_parameters.get("file_url")
        chef = tool_parameters.get("chef")

        # !!<LLM edit>
        # 检查查询参数
        if not query:
            raise ToolProviderCredentialValidationError("查询参数不能为空")
        if not isinstance(query, str):
            raise ToolProviderCredentialValidationError("查询参数必须是字符串类型")
        if len(query.strip()) < 3:
            raise ToolProviderCredentialValidationError("查询参数过短，请提供更具体的查询描述")

        # 检查文件URL参数
        if not file_url:
            raise ToolProviderCredentialValidationError("文件URL不能为空")
        if not isinstance(file_url, str):
            raise ToolProviderCredentialValidationError("文件URL必须是字符串类型")

        # 解析并验证URL
        try:
            parsed_url = urlparse(file_url)
            # 检查URL方案
            if parsed_url.scheme not in ["http", "https"]:
                scheme = parsed_url.scheme or "缺失"
                raise ToolProviderCredentialValidationError(
                    f"无效的URL方案 '{scheme}'。提供的表格文件链接必须以 http:// 或 https:// 开头"
                )

            # 检查URL路径
            if not parsed_url.path or parsed_url.path == "/":
                raise ToolProviderCredentialValidationError("URL缺少有效的文件路径")

            # 检查文件扩展名
            path_parts = parsed_url.path.split("/")
            filename = path_parts[-1] if path_parts else ""
            if not filename or "." not in filename:
                raise ToolProviderCredentialValidationError("URL不包含有效的文件名")

            extension = filename.split(".")[-1].lower()
            valid_extensions = ["csv", "xls", "xlsx"]
            if extension not in valid_extensions:
                raise ToolProviderCredentialValidationError(
                    f"不支持的文件类型: .{extension}。仅支持以下格式: {', '.join(valid_extensions)}"
                )
        except Exception as e:
            if isinstance(e, ToolProviderCredentialValidationError):
                raise
            raise ToolProviderCredentialValidationError(f"URL解析错误: {str(e)}")

        # 检查chef参数
        if not chef:
            raise ToolProviderCredentialValidationError("chef参数不能为空")
        if not isinstance(chef, dict):
            raise ToolProviderCredentialValidationError("chef参数必须是模型对象(Object)")

        # !!</LLM edit>

        ArtifactPayload.validation_prevent_stupidity(chef)

    @classmethod
    def from_s3(cls, tool_parameters: dict[str, Any], *, enable_classifier: bool = True):
        ArtifactPayload.validation_s3(tool_parameters)
        return cls(
            natural_query=tool_parameters.get("query"),
            dify_model_config=tool_parameters.get("chef"),
            table=ArtifactPayload.fetch_table(tool_parameters.get("file_url")),
            enable_classifier=enable_classifier,
        )

    @staticmethod
    def fetch_table(file_url: str) -> File:
        """
        不将文件持久化，该步骤无需严格检查，仅确保 blob 正常获取
        :param file_url:
        :return:
        """
        # 假设 dify-plugin 与目标文件可直连
        headers = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        }
        response = httpx.get(file_url, headers=headers)
        response.raise_for_status()

        # 获取文件名和扩展名
        parsed_url = urlparse(file_url)
        path_parts = parsed_url.path.split("/")
        filename = path_parts[-1] if path_parts else "unknown_file"

        # 提取扩展名
        extension = ""
        if "." in filename:
            extension = f".{filename.split('.')[-1].lower()}"

        # 获取MIME类型
        mime_type = response.headers.get("content-type", None)

        # 获取文件大小
        size = len(response.content)

        # 确定文件类型
        file_type = FileType.DOCUMENT
        if extension in [".csv", ".xls", ".xlsx"]:
            file_type = FileType.DOCUMENT

        return File(
            dify_model_identity=DIFY_FILE_IDENTITY,
            url=file_url,
            mime_type=mime_type,
            filename=filename,
            extension=extension,
            size=size,
            type=file_type,
            _blob=response.content,
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
