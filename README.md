# xiaozhi-update

本文主要修改xiaozhi-server下的search_from_ragflow工具

## **增加功能**

1.rag工具检索中的各项参数， eg. 最低相似度阈值，向量相似度权重，候选chunk数量，最终返回给LLM的chunk数量，关键词混合检索，请求超时，Rerank

2.各项评价的指标，eg.  耗时，问题，质量，返回个数，有效个数命中率，相似度

## 运行与验证

需要依照 https://github.com/xinnan-tech/xiaozhi-esp32-server，https://github.com/infiniflow/ragflow来分别部署xiaozhi-esp32-server和ragflow，再在main\xiaozhi-server\config.yaml中修改相关配置即可

注意：

请至少确保以下配置已填写：

- LLM 的 api_key
- RAGFlow 的 api_key

