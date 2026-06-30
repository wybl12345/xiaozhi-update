import json
import time
import requests
from config.logger import setup_logging
from plugins_func.register import register_function, ToolType, ActionResponse, Action
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.connection import ConnectionHandler

TAG = __name__
logger = setup_logging()

# 定义基础的函数描述模板
SEARCH_FROM_RAGFLOW_FUNCTION_DESC = {
    "type": "function",
    "function": {
        "name": "search_from_ragflow",
        "description": "从知识库中查询信息",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "查询的问题"}},
            "required": ["question"],
        },
    },
}


@register_function(
    "search_from_ragflow", SEARCH_FROM_RAGFLOW_FUNCTION_DESC, ToolType.SYSTEM_CTL
)
def search_from_ragflow(conn: "ConnectionHandler", question=None):
    # 确保字符串参数正确处理编码
    if question and isinstance(question, str):
        # 确保问题参数是UTF-8编码的字符串
        pass
    else:
        question = str(question) if question is not None else ""

    ragflow_config = conn.config.get("plugins", {}).get("search_from_ragflow", {})
    base_url = ragflow_config.get("base_url", "")
    api_key = ragflow_config.get("api_key", "")
    dataset_ids = ragflow_config.get("dataset_ids", [])

    # 检索质量参数（可在 config.yaml 中配置）
    similarity_threshold = ragflow_config.get("similarity_threshold", 0.3)
    vector_similarity_weight = ragflow_config.get("vector_similarity_weight", 0.5)
    top_k = ragflow_config.get("top_k", 1024)
    page_size = ragflow_config.get("page_size", 6)
    keyword = ragflow_config.get("keyword", True)
    request_timeout = ragflow_config.get("timeout", 10)
    rerank_id = ragflow_config.get("rerank_id", "")

    url = base_url + "/api/v1/retrieval"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "question": question,
        "similarity_threshold": similarity_threshold,
        "vector_similarity_weight": vector_similarity_weight,
        "top_k": top_k,
        "page_size": page_size,
        "keyword": keyword,
    }
    if dataset_ids:
        payload["dataset_ids"] = dataset_ids
    if rerank_id:
        payload["rerank_id"] = rerank_id

    logger.bind(tag=TAG).info(
        f"RAGFlow 开始检索 | 问题:「{question}」| "
        f"相似度阈值:{similarity_threshold} | 向量权重:{vector_similarity_weight} | "
        f"关键词混合:{keyword} | 候选数:{top_k} | 返回数:{page_size} | "
        f"Rerank:{rerank_id or '未启用'}"
    )
    start_time = time.time()

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=request_timeout,
            verify=False,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        response.encoding = "utf-8"
        response.raise_for_status()
        result = json.loads(response.text)

        if result.get("code") != 0:
            error_detail = result.get("error", {}).get("detail", "未知错误")
            error_message = result.get("error", {}).get("message", "")
            error_code = result.get("code", "")
            logger.bind(tag=TAG).error(
                f"RAGFlow 业务错误 | 耗时:{elapsed_ms:.0f}ms | "
                f"错误码:{error_code} | 信息:{error_message} | 详情:{error_detail}"
            )

            # 构建详细的错误响应
            error_response = f"RAG接口返回异常（错误码：{error_code}）"

            if error_message:
                error_response += f"：{error_message}"
            if error_detail:
                error_response += f"\n详情：{error_detail}"

            return ActionResponse(Action.RESPONSE, None, error_response)

        chunks_from_api = result.get("data", {}).get("chunks", [])
        # 按综合相似度降序排列（高分在前）
        chunks_from_api = sorted(
            chunks_from_api,
            key=lambda c: c.get("similarity", 0),
            reverse=True,
        )

        contents = []
        valid_chunks_meta = []  # 记录有效 chunk 的元信息，用于评价指标计算
        for chunk in chunks_from_api:
            combined_similarity = chunk.get("similarity", 0)
            # 二次过滤：确保 chunk 相似度达标
            if combined_similarity < similarity_threshold:
                continue
            content = chunk.get("content", "")
            if not content:
                continue
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            else:
                content = str(content)
            doc_name = chunk.get("document_keyword", "")
            vector_sim = chunk.get("vector_similarity", 0)
            term_sim = chunk.get("term_similarity", 0)
            valid_chunks_meta.append({
                # 综合相似度：RAGFlow 已在服务端计算好的最终排序分数
                "similarity": combined_similarity,
                "vector_similarity": vector_sim,
                "term_similarity": term_sim,
                "doc_name": doc_name,
                "preview": content[:50].replace("\n", " "),
            })
            score_info = f"[相关度:{combined_similarity:.2f}" + (f" 来源:{doc_name}" if doc_name else "") + "]"
            contents.append(f"{score_info}\n{content}")

        # ---- 客观评价指标 ----
        total_returned = len(chunks_from_api)   # RAGFlow 实际返回的 chunk 总数
        valid_count = len(valid_chunks_meta)  # 通过阈值过滤后的有效数
        # 命中率 = 有效chunk数 / 返回chunk总数
        hit_rate = valid_count / total_returned if total_returned > 0 else 0.0

        if valid_chunks_meta:
            scores = [m["similarity"] for m in valid_chunks_meta]
            max_score = max(scores)
            min_score = min(scores)
            avg_score = sum(scores) / len(scores)
            # 质量评级
            if max_score >= 0.8:
                quality = "优秀★★★"
            elif max_score >= 0.6:
                quality = "良好★★☆"
            elif max_score >= 0.4:
                quality = "一般★☆☆"
            else:
                quality = "较差☆☆☆"

            logger.bind(tag=TAG).info(
                f"RAGFlow 检索完成 ✓ | 耗时:{elapsed_ms:.0f}ms | 问题:「{question}」| "
                f"质量:{quality} | 返回:{total_returned}个→有效:{valid_count}个 | "
                f"命中率:{hit_rate:.0%} | "
                f"相似度 最高:{max_score:.3f} 均值:{avg_score:.3f} 最低:{min_score:.3f}"
            )
            # 输出每个 chunk 的详细分数（INFO 级别，确保默认配置下可见）
            for idx, m in enumerate(valid_chunks_meta, 1):
                logger.bind(tag=TAG).info(
                    f"  chunk[{idx}] 综合:{m['similarity']:.3f} "
                    f"向量:{m['vector_similarity']:.3f} "
                    f"关键词:{m['term_similarity']:.3f} "
                    f"来源:「{m['doc_name'] or '未知'}」 "
                    f"预览:「{m['preview']}…」"
                )

            context_text = f"# 关于问题【{question}】查到以下知识库内容（按相关度排序）\n"
            context_text += "\n---\n".join(contents)
        else:
            logger.bind(tag=TAG).warning(
                f"RAGFlow 未命中 ✗ | 耗时:{elapsed_ms:.0f}ms | 问题:「{question}」| "
                f"RAGFlow返回:{total_returned}个chunk | 阈值:{similarity_threshold} | "
                f"建议：降低 similarity_threshold 或向知识库补充相关文档"
            )
            context_text = "根据知识库查询结果，没有找到相关信息。"

        return ActionResponse(Action.REQLLM, context_text, None)

    except requests.exceptions.ConnectTimeout:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.bind(tag=TAG).error(
            f"RAGFlow 连接超时 | 耗时:{elapsed_ms:.0f}ms | 超时上限:{request_timeout}s | 地址:{url}"
        )
        return ActionResponse(Action.RESPONSE, None,
            f"RAG接口连接超时（{request_timeout}秒）\n可能原因：RAGFlow服务未启动或网络不通")

    except requests.exceptions.ConnectionError:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.bind(tag=TAG).error(
            f"RAGFlow 连接失败 | 耗时:{elapsed_ms:.0f}ms | 地址:{url}"
        )
        return ActionResponse(Action.RESPONSE, None,
            "无法连接到RAGFlow服务\n可能原因：base_url配置错误或服务未运行")

    except requests.exceptions.Timeout:
        elapsed_ms = (time.time() - start_time) * 1000
        logger.bind(tag=TAG).error(
            f"RAGFlow 请求超时 | 耗时:{elapsed_ms:.0f}ms | 超时上限:{request_timeout}s"
        )
        return ActionResponse(Action.RESPONSE, None,
            f"RAG接口请求超时（{request_timeout}秒）\n建议在config.yaml中增大 timeout 值")

    except requests.exceptions.HTTPError as e:
        elapsed_ms = (time.time() - start_time) * 1000
        status_code = e.response.status_code if hasattr(e, "response") and e.response is not None else "未知"
        logger.bind(tag=TAG).error(
            f"RAGFlow HTTP错误 | 耗时:{elapsed_ms:.0f}ms | 状态码:{status_code}"
        )
        error_response = f"RAG接口HTTP错误（状态码：{status_code}）"
        try:
            detail = e.response.json().get("error", {}).get("message", "")
            if detail:
                error_response += f"\n错误详情：{detail}"
        except Exception:
            pass
        return ActionResponse(Action.RESPONSE, None, error_response)

    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        error_type = type(e).__name__
        logger.bind(tag=TAG).error(
            f"RAGFlow 未知异常 | 耗时:{elapsed_ms:.0f}ms | 类型:{error_type} | 详情:{str(e)}"
        )
        return ActionResponse(Action.RESPONSE, None, f"RAG接口处理异常（{error_type}）：{str(e)}")
