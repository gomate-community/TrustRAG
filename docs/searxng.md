
# SearXNG部署

> https://docs.searxng.org/admin/installation-docker.html#installation-docker

## 部署命令

```bash
docker run --rm \
    --name=searxng \
	-d -p 8080:8080 \
	-v "${PWD}/searxng:/etc/searxng" \
	-e "BASE_URL=http://localhost:8080/" \
	-e "INSTANCE_NAME=my-instance" \
	searxng/searxng
```

>浏览器访问：http://localhost:8080/

## 接口api

可查看源码：[websearch.py](../trustrag/modules/engine/websearch.py)

具体使用方式：

```python

from trustrag.modules.engine.websearch import SearxngEngine, DuckduckEngine

if __name__ == "__main__":
    search_engine = SearxngEngine()
    # Create an instance of SearxngEngine
    searxng = SearxngEngine()

    # Perform a search
    search_results = searxng.search(
        "医疗的梅奥 默沙东",
        language="zh-CN",
        top_k=5
    )

    # Print results
    for result in search_results:
        print(result)
```

## 403错误设置
```text
ValueError: ('Searx API returned an error: ', '<!doctype html>\n\n<title>403 Forbidden</title>\n
```

在`settings.yml`添加json格式
```yaml
  formats:
    - html
    - json
    - csv
    - rss
```

参考：https://github.com/langchain-ai/langchain/issues/855
