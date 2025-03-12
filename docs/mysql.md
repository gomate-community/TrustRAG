## mysql部署

```bash
docker stop mysql

docker rm -f mysql

docker run --name mysql \
    -p 3306:3306 \
    --restart always \
    -v G:/Ubuntu_WSL/rag-middlewares/mysql/data:/var/lib/mysql \
    -e MYSQL_ROOT_PASSWORD=123456 \
    -d mysql:latest

```