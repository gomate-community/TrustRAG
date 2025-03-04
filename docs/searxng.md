
# SearXNG部署

> https://docs.searxng.org/admin/installation-docker.html#installation-docker

## 部署命令

```bash
docker run --rm \
	-d -p 8080:8080 \
	-v "${PWD}/searxng:/etc/searxng" \
	-e "BASE_URL=http://localhost:8080/" \
	-e "INSTANCE_NAME=my-instance" \
	searxng/searxng
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
