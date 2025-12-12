##  选择pytorch镜像

我们可以从dockerhub搜索合适的pytorch镜像，下面为了兼容更多功能，直接选用最新的tag
```bash
docker pull pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
```

## 构建镜像

```bash
docker build -t trustrag:0.1 .
```

```bash
docker build -f ./Dockerfile -t trustrag:1.0 .
```

## 启动容器

```bash
docker run -itd --gpus=all --name=test trustrag:0.1 /bin/bash
```

## 删除容器

```bash
docker rm -f test
```



## 推送阿里云

```bash
docker login --username=11859*****@qq.com registry.cn-beijing.aliyuncs.com
docker tag [ImageId] registry.cn-beijing.aliyuncs.com/quincyqiang/trustrag:[镜像版本号]
docker push registry.cn-beijing.aliyuncs.com/quincyqiang/trustrag:[镜像版本号]
```



要将本地镜像推送到 Harbor 仓库，你需要遵循 Docker 的标准流程：**打标签 (Tag)** -\> **登录 (Login)** -\> **推送 (Push)**。

由于你的仓库地址是 `http` 协议（非 https），可能会遇到 Docker 安全限制的问题，我在下面的步骤中也会说明如何解决。

### 操作步骤

#### 1\. 为镜像打标签 (Tag)

Docker 需要通过镜像的名字知道往哪里推。你需要将本地的 `trustrag:v0.1` 重命名为包含仓库地址和项目名的完整路径。

```bash
docker tag trustrag:v0.1 10.170.130.17/gomate/trustrag:v0.1
```

  * **解释**：
      * `trustrag:v0.1`: 你本地的原始镜像。
      * `10.170.130.17`: Harbor 服务器地址。
      * `gomate`: Harbor 中的**项目名称**。
      * `trustrag`: 仓库名称。
      * `v0.1`: 版本号。

#### 2\. 登录 Harbor (Login)

你需要先登录才能推送。使用你的 Harbor 账号和密码。

```bash
docker login 10.170.130.17
```

  * 输入命令后，按提示输入 Username (用户名) 和 Password (密码)。
  * 如果显示 `Login Succeeded`，则说明登录成功。

#### 3\. 推送镜像 (Push)

登录成功后，执行推送命令：

```bash
docker push 10.170.130.17/gomate/trustrag:v0.1
```

-----

### ⚠️ 关键注意事项 (必读)

#### 1\. 解决 HTTP 协议报错 (server gave HTTP response to HTTPS client)

因为你使用的是 IP 且是 `http` 协议，Docker 默认只允许推送给 HTTPS 安全仓库。如果你在 `login` 或 `push` 时遇到如下错误：

> `http: server gave HTTP response to HTTPS client`

你需要配置 Docker 信任这个不安全的注册表：

1.  **编辑 Docker 配置文件** (Linux 通常在 `/etc/docker/daemon.json`，Windows/Mac 在 Docker Desktop 设置里的 Docker Engine)。
2.  **添加 insecure-registries**：
    ```json
    {
      "insecure-registries": ["10.170.130.17"]
    }
    ```
    *(注意：如果文件里已有其他配置，请确保 JSON 格式正确，每行末尾加逗号)*
3.  **重启 Docker**：
    ```bash
    systemctl restart docker
    ```

#### 2\. 确认项目 `gomate` 存在

在执行 push 之前，请务必登录 Harbor 的网页端，确认**项目 (Project)** 列表中已经有一个名为 **`gomate`** 的项目。

  * 如果 `gomate` 项目不存在，Docker 会报错 `repository ... not found`。
  * 你需要先在网页端新建该项目，并确保该项目是“公开”的，或者你的账号有该项目的“开发者”或“维护者”权限。

-----

**我可以为您做的下一步：**
如果您在执行 `docker login` 时遇到具体的报错信息，可以把报错贴给我，我帮您分析具体原因。