## git常见命令
- 合并分支
```bash
git pull origin main
git merge main
```

- 强制覆盖

```bash
git reset --hard origin/main
```

## 合并PR

这是一个非常经典且常见的 Git 协作场景。你现在的目标是将 `main` 分支上（包含那两个修复安装问题的 PR）的代码，同步到你当前正在工作的 `pipeline` 分支上。

最推荐、最稳妥的方法是使用 **Merge（合并）**。

### 方法一：标准合并 (Recommended)

这种方法最安全，它会把 `main` 上的新内容“合”过来，并保留详细的历史记录。

请在你的终端（当前已经在 `pipeline` 分支）按顺序执行以下步骤：

**1. 切换到 main 分支并更新代码**
首先，你需要确保你本地的 `main` 分支是最新的（包含了你在网页端合并的那两个 PR）。

```bash
git checkout main
git pull origin main
```

**2. 切换回 pipeline 分支**
更新完 `main` 后，回到你的工作分支。

```bash
git checkout pipeline
```

**3. 将 main 合并进 pipeline**
这一步会将 `main` 的修补内容同步到你当前的 `pipeline` 分支。

```bash
git merge main
```

[Image of git merge diagram showing feature branch updating from main]

**4. 处理可能出现的冲突 (Conflict)**

  * **如果一切顺利**：终端会提示 "Merge made by the 'ort' strategy" 或类似信息，你不需要做任何额外操作。
  * **如果有冲突**：Git 会提示 `CONFLICT (content)...`。
      * 你需要打开冲突的文件，找到 `<<<<<<<` 和 `>>>>>>>` 标记的地方，手动修改代码保留你需要的部分。
      * 修改完后，执行 `git add .` 和 `git commit` 来完成合并。

-----

### 方法二：变基 (Rebase) —— 高级选项

如果你希望你的 `pipeline` 分支历史非常干净（看起来就像是你是在那两个 PR 修复**之后**才开始开发 `pipeline` 的），你可以使用 Rebase。

**注意：** 如果 `pipeline` 分支是多人协作的（即别人也在这个分支上写代码），**不要**用这个方法。如果是你自己单独开发的，这个方法更优雅。

```bash
# 1. 更新 main (同上)
git checkout main
git pull origin main

# 2. 回到 pipeline
git checkout pipeline

# 3. 执行变基
git rebase main
```

  * **它的作用**：它会把你在 `pipeline` 上写的代码先“拿下来”，把 `main` 的新代码放进去，然后再把你写的代码“贴”在最后面。这样你的提交历史是一条直线。

-----

### 总结

对于你的情况，**我建议直接使用方法一（Merge）**。

虽然 Rebase 历史好看，但在处理冲突时可能会稍微复杂一点（每一个 commit 都可能冲突）。Merge 是最直观的方式，能立刻让你拥有 `main` 上的修复补丁，继续开发。

**现在的下一步：**
你可以直接在终端试着运行 **方法一** 的命令。如果在第 3 步（merge）遇到了冲突（Conflict）不知道怎么修，你可以把报错信息发给我，我来教你如何解决。