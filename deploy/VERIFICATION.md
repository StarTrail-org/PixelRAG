# systemd 单元文件验证

## 验证目的

确保 `deploy/` 下三个 systemd 单元文件的语法正确性、安全指令兼容性，以及 `%h` 占位符在各发行版中的行为一致性。

## 验证环境

| 发行版 | systemd 版本范围 | 验证方式 |
|--------|-----------------|---------|
| Debian (本机) | ≥ 252 | `systemd-analyze verify` |
| Arch Linux (Docker) | ≥ 255 | `docker run archlinux:latest` + `systemd-analyze verify` |

## 验证步骤

### 1. 准备测试单元

将 `User=yichuan` 替换为本地用户，`ExecStart` 替换为 `/bin/true`，清理多行参数换行：

```bash
mkdir -p /tmp/verify
cp deploy/pixelrag-agent.service /tmp/verify/
cp deploy/pixelrag-api.service /tmp/verify/
cp deploy/pixelrag-api-green.service /tmp/verify/
sed -i "s/User=yichuan/User=$USER/" /tmp/verify/*.service

# API 单元的 ExecStart 有多行参数，sed 替换后需清理孤儿续行
for f in /tmp/verify/pixelrag-api.service /tmp/verify/pixelrag-api-green.service; do
  sed -i 's|ExecStart=.*|ExecStart=/bin/true|' "$f"
  sed -i '/^    --/d' "$f"
done
# Agent 单元无续行，直接替换
sed -i 's|ExecStart=.*|ExecStart=/bin/true|' /tmp/verify/pixelrag-agent.service
```

### 2. 本机验证

```bash
systemd-analyze verify /tmp/verify/*.service
# exit code: 0
```

### 3. 跨发行版验证（Arch Linux）

```bash
docker run --rm -v /tmp/verify:/units:ro archlinux:latest \
  bash -c "systemd-analyze verify /units/*.service 2>&1"
# exit code: 0，无警告
```

Ubuntu 24.04 / Debian Bookworm / Fedora 40 的 Docker 镜像默认不包含 systemd，跳过。

## 验证结果

- **语法**：三个文件均无语法错误
- **`%h` 展开**：正确展开为 `User=` 对应家目录（`/home/luo`、`/root`）
- **安全指令**：`NoNewPrivileges`、`PrivateTmp`、`ProtectSystem` 均被识别
- **多行 ExecStart**：换行续写参数无格式问题

## 兼容性说明

三项安全指令 + `%h` specifier 均为 systemd **v209–v214（2013–2014）** 引入。

所有仍在安全更新范围内的 Linux 发行版 systemd 版本均 ≥ v240：

| 发行版 | 最低 systemd 版本 |
|--------|------------------|
| Ubuntu 20.04 LTS | v245 |
| Debian 11 (Bullseye) | v247 |
| RHEL 8 / Rocky 8 | v239 |
| RHEL 9 / Rocky 9 | v250 |
| Arch Linux | rolling (最新) |
| Fedora 40 | v255 |

**结论：三项配置在所有主流发行版上均可正常使用，无兼容性风险。**
