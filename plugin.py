"""自像比对 (lanyan.self-check) · 主入口

## 它解决什么

机器人分不清「这张像不像我」。之前人格里写了一大段「认图规则」，结果是两件坏事：
① 被问细节就机械应答「认下了」；② 对方一否定就连续改口「行吧行吧…那我认错人了」。
人设里教不出「看」，只能教出「顺从」。所以把那整段删了，换成这条链路。

## 链路

    图片消息到达 → planner.before_request 钩子
      ├ 触发：本轮有图 + 文本含自我指向短语（B 方案，没问到就一次都不花钱）
      ├ 取图：直接从 items 的 UserMessageItem.parts 拿（宿主已带 base64，不用再查库）
      ├ 比对：待判图 + 三视图 → 直连视觉模型，一次请求两张图
      ├ 缓存：按图片字节 md5 存 30 分钟（同一张图反复问不会重复计费）
      └ 注入：往最后一条 UserMessageItem 的尾部追加一行 [自像比对]

## 为什么是「注入」而不是「工具」

`search_self_information` 零调用、`view_all_image` 零调用——要调「这像不像我」的工具，
planner 得先意识到「这可能是我」，而这正是模型做不好的判断。触发不能依赖模型起意。

## 为什么是「请求注记」而不是「写图片描述」

图片描述落在 `images.description`，会进检索、摘要、长期记忆。把一次判读（可能判错）
写进去 = 把错误固化成记忆，而且按图 hash 缓存后改不回来。请求注记只作用于这一轮。

## 边界

- 不改数据库、不发消息、不写图片描述、不动人格。
- 任何失败（超时/无密钥/解码失败/输出非法）都只是不注入，绝不影响正常回话。
- 直连不进宿主 llm_usage 账本，自己在 data 目录记 `cost.jsonl` 便于核算。
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from maibot_sdk import HookHandler, MaiBotPlugin
from maibot_sdk.types import HookMode

from .config import SelfCheckConfig

try:  # Pillow 是可选依赖：缺了就不缩放，直接用原图（费点 token，但不至于不能用）
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

_VERDICTS = ("same", "similar", "different", "unclear")


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """从模型回复里抠出 JSON。容忍 ``` 围栏和前后废话。"""

    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    candidates: List[str] = []
    if fence:
        candidates.append(fence.group(1).strip())
    brace = re.search(r"\{.+?\}", text, re.S)
    if brace:
        candidates.append(brace.group(0))
    candidates.append(text)
    for item in candidates:
        try:
            data = json.loads(item)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


# items 里的聊天记录渲染格式：<message msg_id="…" time="…" user="…" …>正文</message>。
# 🔴 实测教训：原正则把属性顺序写死了（msg_id→time→user 必须连着），
# 宿主只要换个顺序就整条链路静默失效——钩子跑完一条日志都不留。所以改成
# 「先抓标签体，再从属性串里逐个取键」，属性顺序、多出来的属性都不影响。
_MESSAGE_TAG_RE = re.compile(r"<message\b(?P<attrs>[^>]*)>(?P<body>.*?)</message>", re.S)
_ATTR_RE = re.compile(r'(?P<key>[A-Za-z_][\w-]*)\s*=\s*"(?P<val>[^"]*)"')


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _decode_b64(encoded: Any) -> Optional[bytes]:
    """解 base64（兼容 data:image/... 前缀）。"""

    if not isinstance(encoded, str) or not encoded:
        return None
    if encoded.startswith("data:"):
        match = re.match(r"^data:image/[^;,]+;base64,(.*)$", encoded, flags=re.I | re.S)
        if not match:
            return None
        encoded = match.group(1)
    try:
        return base64.b64decode(encoded, validate=False) or None
    except Exception:
        return None


def _unwrap_message(payload: Any) -> Any:
    """宿主可能直接返回消息字典，也可能包一层 {"message": …}，两种都要认。

    🔴 实测：message.get_by_id 回来的是 {"message_id": …, "raw_message": […]}，
    没有外层 message 键——若按 payload["message"] 取值会拿到 None。
    """

    if isinstance(payload, Mapping) and isinstance(payload.get("message"), Mapping):
        return payload["message"]
    return payload


def _extract_images(message: Any) -> List[bytes]:
    """从消息原始结构里取图片 bytes。"""

    if not isinstance(message, Mapping):
        return []
    segments: List[Any] = []
    for key in ("raw_message", "segments", "message_segments", "content"):
        value = message.get(key)
        if isinstance(value, list):
            segments.extend(value)
    images: List[bytes] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        kind = str(segment.get("type") or segment.get("content_type") or "").casefold()
        if kind not in {"image", "picture", "photo"}:
            continue
        encoded = _first_text(
            segment.get("binary_data_base64"),
            segment.get("base64"),
            segment.get("data_base64"),
        )
        if not encoded:
            data = segment.get("data")
            if isinstance(data, str) and data.startswith("data:image/"):
                encoded = data
        image = _decode_b64(encoded)
        if image:
            images.append(image)
    return images


class LanyanSelfCheckPlugin(MaiBotPlugin):
    """把「像不像她」变成一次客观比对，再交给 planner 去说。"""

    config_model = SelfCheckConfig

    def __init__(self) -> None:
        super().__init__()
        self._plugin_dir = os.path.dirname(os.path.abspath(__file__))
        # 图片 hash -> (monotonic_expire, result)
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        # 「同一句话 + 同一张图」的去重闸门：planner 一条消息会连调好几轮，
        # 不去重就会把同一行注记塞进去好几遍，她会把「不是我」重复说三次。
        self._last_note: Dict[str, float] = {}

    # ---------------- 生命周期 ----------------

    async def on_load(self) -> None:
        cfg = self._cfg()
        ref = self._ref_path()
        probe = str(getattr(cfg.debug, "probe_msg_id", "") or "").strip()
        if probe:
            data = await self._image_by_id("", probe)
            self.ctx.logger.info(
                "[SelfCheck] 探针：msg=%s 取到图=%s 字节=%s", probe, bool(data), len(data or b"")
            )
            if data:
                ref = self._ref_path()
                result = await self._compare(data, ref) if ref else None
                self.ctx.logger.info(
                    "[SelfCheck] 探针比对: %s",
                    json.dumps(result, ensure_ascii=False, default=str)[:300] if result else "失败",
                )
        self.ctx.logger.info(
            "自像比对 v%s 已加载（启用=%s，模型=%s，基准图=%s，日上限=%s，缓存=%s 分钟）",
            cfg.plugin.version,
            cfg.plugin.enabled,
            cfg.engine.model,
            "有" if ref else "缺失",
            cfg.limits.daily_max or "不限",
            cfg.limits.cache_minutes,
        )

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        del config_data, version
        self._cache.clear()
        self.ctx.logger.info("自像比对配置更新: scope=%s（缓存已清）", scope)

    async def on_unload(self) -> None:
        self._cache.clear()
        self._last_note.clear()
        self.ctx.logger.info("自像比对已卸载")

    # ---------------- 基础 ----------------

    def _cfg(self) -> SelfCheckConfig:
        return self.config  # type: ignore[return-value]

    def _data_dir(self) -> str:
        try:
            base = str(self.ctx.paths.data_dir)
        except Exception:
            base = os.path.join(self._plugin_dir, "data")
        os.makedirs(base, exist_ok=True)
        return base

    def _instance_root(self) -> str:
        """plugins/lanyan_self_check -> 实例的 modules/MaiBot。"""

        return os.path.dirname(os.path.dirname(self._plugin_dir))

    # ---------------- 参考图 ----------------

    def _resolve(self, path: str) -> str:
        if not path:
            return ""
        return path if os.path.isabs(path) else os.path.join(self._plugin_dir, path)

    def _ref_path(self) -> str:
        cfg = self._cfg()
        primary = self._resolve(cfg.refs.primary)
        if primary and os.path.isfile(primary):
            return primary
        fallback = self._resolve(cfg.refs.fallback)
        if fallback and os.path.isfile(fallback):
            return fallback
        return ""

    # ---------------- 额度 ----------------

    def _quota_path(self) -> str:
        return os.path.join(self._data_dir(), "daily.json")

    def _quota_take(self) -> bool:
        """扣一次日额度。上限 <= 0 表示不限。"""

        limit = int(self._cfg().limits.daily_max or 0)
        if limit <= 0:
            return True
        today = time.strftime("%Y-%m-%d")
        path = self._quota_path()
        state: Dict[str, Any] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception:
            state = {}
        if not isinstance(state, dict) or state.get("date") != today:
            state = {"date": today, "count": 0}
        used = int(state.get("count") or 0)
        if used >= limit:
            return False
        state["count"] = used + 1
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(state, handle, ensure_ascii=False)
        except Exception:
            pass
        return True

    def _record_cost(self, row: Dict[str, Any]) -> None:
        try:
            with open(os.path.join(self._data_dir(), "cost.jsonl"), "a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---------------- 触发与取图 ----------------

    @staticmethod
    def _user_items(items: Any) -> List[Dict[str, Any]]:
        if not isinstance(items, list):
            return []
        return [it for it in items if isinstance(it, dict) and it.get("item_type") == "UserMessageItem"]

    @staticmethod
    def _parts(item: Mapping[str, Any]) -> List[Dict[str, Any]]:
        parts = item.get("parts")
        return [p for p in parts if isinstance(p, dict)] if isinstance(parts, list) else []

    @classmethod
    def _text_of(cls, item: Mapping[str, Any]) -> str:
        chunks: List[str] = []
        for part in cls._parts(item):
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "\n".join(chunks).strip()

    @classmethod
    def _recent_user_messages(cls, items: Any, lookback: int) -> List[Dict[str, Any]]:
        """兜底路径：从 Context Items 的文本里解析出真实用户消息。

        🔴 items 里**没有图片二进制**——宿主把图换成了「[图片：…]」描述文本，
        所以取图只能靠这里拿到的 msg_id 去消息接口要原图。

        只在 message.get_recent 拿不到东西时才走这条路（见 on_planner_before_request）。
        """

        messages: List[Dict[str, Any]] = []
        for item in cls._user_items(items):
            for tag in _MESSAGE_TAG_RE.finditer(cls._text_of(item)):
                attrs = {
                    m.group("key").casefold(): m.group("val")
                    for m in _ATTR_RE.finditer(tag.group("attrs") or "")
                }
                if str(attrs.get("is_self_message") or "").casefold() == "true":
                    continue
                body = (tag.group("body") or "").strip()
                msg_id = str(attrs.get("msg_id") or attrs.get("id") or "").strip()
                if not body or not msg_id:
                    continue
                messages.append(
                    {
                        "id": msg_id,
                        "time": str(attrs.get("time") or ""),
                        "text": body,
                        "has_image": "图片" in body or "[图" in body,
                        "ts": cls._parse_ts(attrs.get("time")),
                    }
                )
        return messages[-max(1, lookback):]

    # ---- 宿主消息接口（首选路径） ----

    @staticmethod
    def _rows_of(payload: Any) -> List[Dict[str, Any]]:
        """把 message.get_recent 的几种返回形态统一成字典列表。"""

        if isinstance(payload, Mapping):
            for key in ("messages", "message", "data", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(r) for r in value if isinstance(r, Mapping)]
                if isinstance(value, Mapping):
                    return [dict(value)]
            return []
        if isinstance(payload, list):
            return [dict(r) for r in payload if isinstance(r, Mapping)]
        return []

    @staticmethod
    def _row_is_self(row: Mapping[str, Any]) -> bool:
        """判断是不是她自己发的。

        🔴 消息字典没有顶层 user_id——用户 id 埋在 message_info.user_info.user_id。
        """

        info = row.get("message_info")
        user_id = ""
        if isinstance(info, Mapping):
            user = info.get("user_info")
            if isinstance(user, Mapping):
                user_id = str(user.get("user_id") or "")
        account = str(row.get("account_id") or "")
        return bool(user_id and account and user_id == account)

    @staticmethod
    def _row_text(row: Mapping[str, Any]) -> str:
        plain = row.get("processed_plain_text")
        if isinstance(plain, str) and plain.strip():
            return plain.strip()
        chunks: List[str] = []
        for key in ("raw_message", "segments", "content"):
            value = row.get(key)
            if not isinstance(value, list):
                continue
            for segment in value:
                if not isinstance(segment, Mapping):
                    continue
                kind = str(segment.get("type") or segment.get("content_type") or "").casefold()
                if kind not in {"text", "plain"}:
                    continue
                data = segment.get("data")
                if isinstance(data, str) and data.strip():
                    chunks.append(data.strip())
        return "\n".join(chunks).strip()

    @staticmethod
    def _row_has_image(row: Mapping[str, Any]) -> bool:
        for key in ("raw_message", "segments", "content"):
            value = row.get(key)
            if not isinstance(value, list):
                continue
            for segment in value:
                if not isinstance(segment, Mapping):
                    continue
                kind = str(segment.get("type") or segment.get("content_type") or "").casefold()
                if kind in {"image", "picture", "photo"}:
                    return True
        return False

    @staticmethod
    def _row_group(row: Mapping[str, Any]) -> str:
        """群号。私聊返回空串。"""

        info = row.get("message_info")
        if isinstance(info, Mapping):
            group = info.get("group_info")
            if isinstance(group, Mapping) and group.get("group_id"):
                return str(group.get("group_id"))
        return ""

    @staticmethod
    def _row_at(row: Mapping[str, Any]) -> bool:
        """有没有 @ 她 / 有没有叫她的名字或别名。

        🔴 不用自己扫词：宿主在入库时已经算好 is_at / is_mentioned，
        其中 is_mentioned 就是拿 bot_config 的 nickname + alias_names 扫的文本。
        实测群消息 4316 条里 20 条命中，字段是可靠的。
        """

        return bool(row.get("is_at")) or bool(row.get("is_mentioned"))

    @staticmethod
    def _parse_ts(value: Any) -> float:
        """把各种时间写法统一成秒级 Unix 时间戳；认不出返回 0。

        宿主 message.get_recent 给的是 ``str(timestamp.timestamp())``（秒，带小数）；
        兜底解析 items 时拿到的是 ``"11-09 21:43:28"``（不带年份）。
        """

        if value is None or value == "":
            return 0.0
        try:
            num = float(value)
        except (TypeError, ValueError):
            num = 0.0
        if num > 1e11:  # 毫秒
            num /= 1000.0
        if num > 1e9:  # 2001 年之后，当作合法时间戳
            return num
        text = str(value).strip()
        for fmt in ("%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                mark = datetime.strptime(text, fmt)
            except ValueError:
                continue
            if mark.year == 1900:  # %m-%d 不带年，补成今年
                mark = mark.replace(year=datetime.now().year)
            return mark.timestamp()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0

    @classmethod
    def _row_ts(cls, row: Mapping[str, Any]) -> float:
        """这一条消息的发送时间（秒）。取不到返回 0，调用方按「不限时间」处理。"""

        for key in ("timestamp", "received_at", "time", "ts"):
            ts = cls._parse_ts(row.get(key))
            if ts > 0:
                return ts
        return 0.0

    @staticmethod
    def _group_gate(text_msg: Mapping[str, Any], require_at: bool) -> bool:
        """这一条能不能触发比对（True = 放行）。

        私聊一律放行；群聊在 require_at 打开时，要求 @ 了她或直接叫了她的名字/别名。
        """

        if not require_at:
            return True
        if not text_msg.get("group"):
            return True
        return bool(text_msg.get("at"))

    async def _recent_rows(self, session: str, limit: int) -> List[Dict[str, Any]]:
        """按会话取最近消息（不带二进制，轻量）。

        命中关键词后才会按 msg_id 去取那一张图的字节——避免每轮规划都拖几 MB base64。
        """

        if not session:
            return []
        try:
            payload = await self.ctx.call_capability(
                "message.get_recent", chat_id=session, limit=limit
            )
        except Exception as exc:
            # 🔴 原来是 debug：钩子整条静默消失，日志里一个字都没有，无从定位。
            # 21:23 那次「私聊发图 + 左边那个是你吗」没触发，就是卡在这种静默出口上。
            self.ctx.logger.warning(
                "[SelfCheck] 取最近消息失败: session=%s %s: %s",
                session[:12],
                type(exc).__name__,
                exc,
            )
            return []
        return self._rows_of(payload)

    def _should_check(self, text: str) -> bool:
        cfg = self._cfg()
        body = (text or "").casefold()
        if not body:
            return False
        for keyword in cfg.trigger.keywords or []:
            token = str(keyword or "").strip().casefold()
            if token and token in body:
                return True
        return False

    @staticmethod
    def _select_question(
        messages: List[Dict[str, Any]],
        is_hit: Callable[[str], bool],
        lookback: int,
        max_age: float,
    ) -> Tuple[Dict[str, Any], str, str]:
        """从最近几条里挑出「这一轮要判的那句问话」。

        返回 ``(命中的消息, 命中文本, 最近一句文本)``；没命中时前两项为空。

        🔴 为什么不能只取最后一条文本：实测先发图、紧接着问「这是你吗」、
        过一会儿又补一句「在？」，最后一条不含关键词，
        整轮静默跳过，而真正要判的那句就在前面。

        两道闸防「翻旧账」：
        ① 只在最近 ``lookback`` 条里找（默认 4）——几分钟前问过的话不该再翻出来；
        ② 命中句与最新一条消息的间隔不得超过 ``max_age`` 秒（默认 120）——
           实测两句相隔约 84 秒，窗口设 60 会正好漏掉。
        """

        window = messages[-max(1, int(lookback or 1)):]
        latest_ts = 0.0
        for message in reversed(messages):
            latest_ts = float(message.get("ts") or 0)
            if latest_ts > 0:
                break
        if latest_ts <= 0:
            latest_ts = time.time()

        latest_text = ""
        for message in reversed(window):
            body = str(message.get("text") or "")
            if not body:
                continue
            if not latest_text:
                latest_text = body
            if not is_hit(body):
                continue
            ts = float(message.get("ts") or 0)
            # ts 取不到（0）时不做时间判断，只靠条数兜底，宁可多判一次也别静默漏掉
            if max_age > 0 and ts > 0 and latest_ts - ts > max_age:
                continue
            return message, body, latest_text
        return {}, "", latest_text

    # ---------------- 图片编码 ----------------

    def _to_data_uri(self, data: bytes, max_side: int) -> Optional[str]:
        cfg = self._cfg()
        if Image is None:
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()
        try:
            image = Image.open(io.BytesIO(data))
            image = image.convert("RGB")
            if max_side > 0:
                image.thumbnail((int(max_side), int(max_side)))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(cfg.engine.jpeg_quality or 85))
            return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        except Exception as exc:
            self.ctx.logger.warning("自像比对：图片解码失败（%s），跳过", exc)
            return None

    # ---------------- 引擎 ----------------

    def _provider(self) -> Tuple[str, str]:
        eng = self._cfg().engine
        if eng.base_url and eng.api_key:
            return eng.base_url.rstrip("/"), eng.api_key
        path = os.path.join(self._instance_root(), "config", "model_config.toml")
        try:
            import tomllib

            with open(path, "rb") as handle:
                data = tomllib.load(handle)
            for provider in data.get("api_providers") or []:
                if str(provider.get("name") or "") == eng.provider_name:
                    base = str(provider.get("base_url") or "").rstrip("/")
                    key = str(provider.get("api_key") or "")
                    if base and key:
                        return base, key
        except Exception as exc:
            self.ctx.logger.warning("自像比对：读 model_config.toml 失败（%s）", exc)
        return "", ""

    def _call_engine(self, payload: Dict[str, Any], base: str, key: str, timeout: float) -> Dict[str, Any]:
        """同步 HTTP（放线程里跑）。返回原始响应 JSON。"""

        import urllib.request

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            base + "/chat/completions",
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))

    async def _compare(self, image: bytes, ref_path: str) -> Optional[Dict[str, Any]]:
        cfg = self._cfg()
        base, key = self._provider()
        if not base or not key:
            self.ctx.logger.warning("自像比对：没有可用的接口/密钥，跳过")
            return None
        try:
            with open(ref_path, "rb") as handle:
                ref_bytes = handle.read()
        except Exception as exc:
            self.ctx.logger.warning("自像比对：基准图读取失败（%s），跳过", exc)
            return None
        ref_uri = self._to_data_uri(ref_bytes, int(cfg.engine.max_ref_side or 1024))
        test_uri = self._to_data_uri(image, int(cfg.engine.max_test_side or 896))
        if not ref_uri or not test_uri:
            return None
        payload = {
            "model": cfg.engine.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": (cfg.prompt.compare or "").strip()},
                        {"type": "image_url", "image_url": {"url": ref_uri}},
                        {"type": "image_url", "image_url": {"url": test_uri}},
                    ],
                }
            ],
            "temperature": float(cfg.engine.temperature or 0.1),
            "max_tokens": int(cfg.engine.max_tokens or 300),
        }
        started = time.time()
        response = await asyncio.to_thread(
            self._call_engine, payload, base, key, float(cfg.engine.timeout_seconds or 25.0)
        )
        latency = time.time() - started
        choices = response.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            content = str((choices[0].get("message") or {}).get("content") or "")
        data = _extract_json(content)
        if not data:
            self.ctx.logger.info("自像比对：输出无法解析（%s）", content[:120])
            return None
        verdict = str(data.get("verdict") or "").strip().lower()
        if verdict not in _VERDICTS:
            self.ctx.logger.info("自像比对：verdict 非法（%r）", verdict)
            return None
        try:
            confidence = float(data.get("confidence"))
        except Exception:
            confidence = 0.0
        differences = data.get("differences")
        if not isinstance(differences, list):
            differences = []
        differences = [str(x).strip() for x in differences if str(x).strip()][:3]
        result = {
            "verdict": verdict,
            "confidence": max(0.0, min(1.0, confidence)),
            "differences": differences,
            "one_line": str(data.get("one_line") or "").strip(),
        }
        usage = _as_dict(response.get("usage"))
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        result["_cost"] = (prompt_tokens * 0.7 + completion_tokens * 2.8) / 1e6
        result["_latency"] = latency
        result["_tokens"] = (prompt_tokens, completion_tokens)
        return result

    # ---------------- 注入文案 ----------------

    def _verdict_label(self, verdict: str) -> str:
        raw = self._cfg().inject.verdict_labels or ""
        for chunk in raw.split("|"):
            name, _, label = chunk.partition("=")
            if name.strip() == verdict and label.strip():
                return label.strip()
        return verdict

    def _advice(self, verdict: str, differences: List[str]) -> str:
        """按结论给一句行为要求。

        🔴 实测：光给结论不够。判了 different 之后模型仍会
        「我也不是干不出来」「我倒是认了」——这种含糊口吻读起来就是在认领。
        所以不同档位要分别要求「先怎么说」。
        """

        if verdict == "different":
            return (
                "这是客观比对结果，不是她的印象。回话时第一句就明确撇清（「不是我」），再顺带聊图；"
                "不许用「我也不是干不出来」「我倒是认了」「行吧那是我」这类含糊说法，那等于认领。"
            )
        if verdict == "same":
            return (
                "这是客观比对结果，已经确认是同一个人，不用自己再核一遍细节。像就直接认下来，"
                "可以顺带吐槽一句；别否认、别改口。AI 出图常把特征画丢（主特征变淡或没画、配饰遗漏、"
                "瞳色偏色），这些不改变结论，不许拿细节偏差当理由否认；"
                "也不许刚认完又补一条推翻自己。图里有多个人时别瞎指「左边/右边那个是我」，"
                "不确定就只说「图里有个人像我」。"
            )
        if verdict == "similar":
            detail = "、".join(differences) or "有出入的地方"
            return f"这是客观比对结果。先说像，再点出一处具体差别（{detail}），别含糊两可、也别整段否认。"
        return "这是客观比对结果，但图看不清。看不清就说看不清，别猜、别硬认。"

    def _note(self, result: Mapping[str, Any]) -> str:
        cfg = self._cfg()
        differences = [str(x) for x in (result.get("differences") or [])]
        if differences:
            diff_line = "外貌差异：" + "／".join(differences) + "。"
        else:
            diff_line = "外貌差异：无（服装、场景、画风不计）。"
        template = cfg.inject.template or ""
        try:
            return template.format(
                verdict_cn=self._verdict_label(str(result.get("verdict"))),
                verdict=result.get("verdict"),
                confidence=f"{float(result.get('confidence') or 0):.2f}",
                differences="、".join(differences) or "无",
                diff_line=diff_line,
                one_line=result.get("one_line") or "",
                advice=self._advice(str(result.get("verdict")), differences),
            ).strip()
        except Exception as exc:
            # 🔴 模板渲染失败绝不能静默：那样会退回「只有结论、没有行为要求」的简版，
            # 正是 v0.2.1 要治的毛病，而且日志上完全看不出来。
            self.ctx.logger.warning(
                "[SelfCheck] 注入模板渲染失败（%s），已降级为简版注记。template=%r",
                exc,
                template[:80],
            )
            return (
                f"[自像比对] 已比对：{self._verdict_label(str(result.get('verdict')))}，"
                f"置信 {float(result.get('confidence') or 0):.2f}。{diff_line}"
                f"{self._advice(str(result.get('verdict')), differences)}"
            )

    @staticmethod
    def _append_note(items: List[Any], note: str) -> bool:
        """往最后一条 UserMessageItem 的尾部追加，不碰 system（前缀缓存）。"""

        for item in reversed(items):
            if not isinstance(item, dict) or item.get("item_type") != "UserMessageItem":
                continue
            parts = item.get("parts")
            if not isinstance(parts, list) or not parts:
                continue
            for part in reversed(parts):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = f"{part.get('text') or ''}\n{note}"
                    return True
            parts.append({"type": "text", "text": note})
            return True
        return False

    # ---------------- 钩子 ----------------

    @HookHandler(
        "maisaka.planner.before_request",
        mode=HookMode.BLOCKING,
        # 宿主 HookSpec 的默认预算是 6000ms，但本插件单次比对实测 6.8s，
        # 且内部 wait_for 上限是 timeout_seconds + 3 = 18s。预算小于 18s 时，
        # host 会先杀协程：插件的「比对超时，本轮不注入」优雅兜底永远触发不到，
        # 慢请求反被判 HookHandler 超时计入熔断（连续 3 次熔断 60s 起）。
        # 已实证：宿主规划超时短于视觉模型耗时时，插件即使跑完并打出「比对完成」，
        # 注入也已被丢弃，那次调用白费。
        # 故在组件级声明 20000ms 覆盖 spec 默认值（优先级 target > spec > 全局），
        # 宿主源码零改动。取值 = 插件上限 18s + 2s 余量（RPC 序列化与调度）。
        timeout_ms=20000,
    )
    async def on_planner_before_request(self, **kwargs: Any) -> Dict[str, Any]:
        cfg = self._cfg()
        if not cfg.plugin.enabled:
            return {"action": "continue"}
        items = kwargs.get("items")
        session = str(kwargs.get("session_id") or "")
        lookback = int(cfg.trigger.lookback or 4)
        trace = bool(getattr(cfg.debug, "trace", False))

        messages = await self._collect_messages(session, items, lookback, trace)
        if not messages:
            if trace:
                self.ctx.logger.info("[SelfCheck] trace: 拿不到用户消息，跳过")
            return {"action": "continue"}

        text_lookback = int(getattr(cfg.trigger, "text_lookback", 4) or 4)
        max_age = float(getattr(cfg.trigger, "text_max_age_seconds", 120) or 0)
        text_msg, text, latest_text = self._select_question(
            messages, self._should_check, text_lookback, max_age
        )
        if not text:
            if trace:
                self.ctx.logger.info(
                    "[SelfCheck] trace: 最近 %d 条里没有命中关键词的问句（最新一句=%r），跳过",
                    text_lookback,
                    latest_text[:40],
                )
            return {"action": "continue"}

        # 🔴 群聊闸门：必须 @ 了她、或直接叫了她的名字/别名，才比对。
        # 群里是一堆人说话，别人之间「这是你吗」并不指向她，
        # 不拦的话会拿别人刚发的图去判，她可能突然插一句「不是我」。
        if not self._group_gate(text_msg, bool(getattr(cfg.trigger, "group_require_at", True))):
            if trace:
                self.ctx.logger.info(
                    "[SelfCheck] trace: 群聊但没 @ 也没叫名字，跳过 text=%r", text[:30]
                )
            return {"action": "continue"}

        image, image_id = await self._pick_image(session, messages)
        if image is None:
            # 🔴 诊断出口 3/3：命中了关键词却拿不到图，原来只有 debug，等同静默。
            # has_image 是靠 processed_plain_text 里的「[图片」判断的，标记失真也会落到这里。
            self.ctx.logger.info(
                "[SelfCheck] 命中关键词但没取到图: text=%r 看了 %d 条（标了图的 %d 条）",
                text[:30],
                len(messages),
                sum(1 for m in messages if m.get("has_image")),
            )
            return {"action": "continue"}
        self.ctx.logger.info(
            "[SelfCheck] 触发: text=%r 图源=%s 候选=%d 条",
            text[:30], image_id or "recent", len(messages),
        )

        digest = hashlib.md5(image).hexdigest()

        # 🔴 同一句话 + 同一张图，注入过就不再注入。
        # 实测 planner 为一条用户消息会连调 2~3 轮，每轮都注入就等于让她把结论重复说几遍。
        # 原来只挡 30 秒，但用户常常隔一分多钟才补一句「在？」（实测 84 秒），
        # 窗口太短会让同一句问话被答两遍。这里跟图片缓存对齐（默认 30 分钟）：
        # 缓存还没过期就说明结论没变，没必要再说一次。
        dedupe_sec = max(30.0, float(cfg.limits.cache_minutes or 0) * 60.0)
        gate = hashlib.md5((digest + "\x00" + text).encode("utf-8")).hexdigest()
        last_at = self._last_note.get(gate)
        if last_at is not None and time.monotonic() - last_at < dedupe_sec:
            if trace:
                self.ctx.logger.info(
                    "[SelfCheck] trace: 这句问话 %.0f 分钟内已注入过，跳过重复注入",
                    dedupe_sec / 60.0,
                )
            return {"action": "continue"}

        cached = self._cache.get(digest)
        now = time.monotonic()
        result: Optional[Dict[str, Any]] = None
        from_cache = False
        if cached and cached[0] > now:
            result = cached[1]
            from_cache = True
            self.ctx.logger.debug("[SelfCheck] 命中缓存: %s", digest[:10])

        if result is None:
            if not self._quota_take():
                self.ctx.logger.info("[SelfCheck] 今日比对已达上限（%s），跳过", cfg.limits.daily_max)
                return {"action": "continue"}
            ref_path = self._ref_path()
            if not ref_path:
                self.ctx.logger.warning("[SelfCheck] 没有可用的基准图，跳过")
                return {"action": "continue"}
            try:
                result = await asyncio.wait_for(
                    self._compare(image, ref_path),
                    timeout=float(cfg.engine.timeout_seconds or 25.0) + 3.0,
                )
            except asyncio.TimeoutError:
                self.ctx.logger.info("[SelfCheck] 比对超时，本轮不注入")
                return {"action": "continue"}
            except Exception as exc:
                self.ctx.logger.warning("[SelfCheck] 比对失败，本轮不注入: %s", exc)
                return {"action": "continue"}
            if result is None:
                return {"action": "continue"}
            minutes = int(cfg.limits.cache_minutes or 0)
            if minutes > 0:
                self._cache[digest] = (now + minutes * 60, result)

        verdict = str(result.get("verdict"))
        confidence = float(result.get("confidence") or 0)
        note = self._note(result)
        if not self._append_note(items, note):
            return {"action": "continue"}
        # 注入成功了才上闸；比对失败/超时不上，下一轮还能重试。
        self._last_note[gate] = time.monotonic()
        # 去重窗口从 30 秒拉到 30 分钟后，字典会慢慢变长，攒够 200 条清一次过期项。
        if len(self._last_note) > 200:
            cutoff = time.monotonic() - dedupe_sec
            for stale in [k for k, v in self._last_note.items() if v < cutoff]:
                self._last_note.pop(stale, None)

        latency = float(result.get("_latency") or 0)
        self.ctx.logger.info(
            "[SelfCheck] 比对完成: verdict=%s 置信=%.2f 差异=%s 耗时=%.1fs 估算成本=%.4f%s",
            verdict,
            confidence,
            "/".join(result.get("differences") or []) or "无",
            latency,
            0.0 if from_cache else float(result.get("_cost") or 0),
            "（缓存复用，不再计费）" if from_cache else "",
        )
        # 🔴 只有真调了模型才记账。缓存复用的结论照抄没花钱，
        # 实测曾把一次调用记成六次，cost.jsonl 虚报六倍，所以这里按轮次去重。
        if not from_cache:
            self._record_cost(
                {
                    "ts": int(time.time()),
                    "session": str(kwargs.get("session_id") or ""),
                    "hash": digest[:16],
                    "verdict": verdict,
                    "confidence": round(confidence, 3),
                    "differences": result.get("differences") or [],
                    "latency": round(latency, 2),
                    "cost": round(float(result.get("_cost") or 0), 6),
                    "tokens": list(result.get("_tokens") or []),
                }
            )
        kwargs["items"] = items
        return {"action": "continue", "modified_kwargs": kwargs}

    async def _image_by_id(self, stream_id: str, message_id: str) -> Optional[bytes]:
        """按 msg_id 取原图。这是唯一能拿到图片字节的路（items 里只有描述文本）。"""

        if not message_id:
            return None
        # 先带会话 id 试（业务上更精确）；不行再不带——msg_id 通常全局唯一，
        # 而 hook 给的 session_id 未必等于消息的 chat_id，带错了反而查不到。
        attempts: List[Dict[str, str]] = [{"stream_id": stream_id}] if stream_id else []
        attempts.append({})
        for extra in attempts:
            try:
                payload = await self.ctx.message.get_by_id(
                    message_id, include_binary_data=True, **extra
                )
            except Exception as exc:
                self.ctx.logger.debug("[SelfCheck] 按 ID 取图失败 %s %s: %s", message_id, extra, exc)
                continue
            images = _extract_images(_unwrap_message(payload))
            if images:
                return images[0]
        return None

    async def _collect_messages(
        self, session: str, items: Any, lookback: int, trace: bool
    ) -> List[Dict[str, Any]]:
        """拿到「最近几条真实用户消息」，带 msg_id 与是否有图。

        首选宿主消息接口（结构确定），拿不到才退回解析 items 文本。
        """

        rows = await self._recent_rows(session, max(lookback * 5, 24))
        if not rows:
            # 🔴 诊断出口 1/3：会话接口一条都没给（session 为空 / capability 报错 / chat_id 对不上）。
            # 原来这条路径完全静默，只能靠猜。
            self.ctx.logger.info(
                "[SelfCheck] 会话取不到消息: session=%s items=%s，转 items 兜底解析",
                session[:12] or "(空)",
                len(items) if isinstance(items, list) else type(items).__name__,
            )
        if trace:
            self.ctx.logger.info(
                "[SelfCheck] trace: session=%s items=%s rows=%d",
                session[:12],
                len(items) if isinstance(items, list) else "None",
                len(rows),
            )

        messages: List[Dict[str, Any]] = []
        for row in rows:
            if self._row_is_self(row):
                continue
            msg_id = str(row.get("message_id") or row.get("msg_id") or "").strip()
            if not msg_id:
                continue
            body = self._row_text(row)
            has_image = self._row_has_image(row)
            if not body and not has_image:
                continue
            messages.append(
                {
                    "id": msg_id,
                    "text": body,
                    "has_image": has_image,
                    "group": bool(self._row_group(row)),
                    "at": self._row_at(row),
                    "ts": self._row_ts(row),
                }
            )

        if not messages:
            fallback = self._recent_user_messages(items, lookback)
            if trace:
                self.ctx.logger.info("[SelfCheck] trace: 退回 items 解析得到 %d 条", len(fallback))
            messages = fallback
            if not messages:
                # 🔴 诊断出口 2/3：两条路都空 ⇒ 钩子会无声地 return，调用方什么也看不到。
                self.ctx.logger.info(
                    "[SelfCheck] 没拿到用户消息: session=%s rows=%d items=%s，跳过本轮",
                    session[:12] or "(空)",
                    len(rows),
                    len(items) if isinstance(items, list) else type(items).__name__,
                )

        return messages[-max(1, lookback):]

    async def _pick_image(
        self, stream_id: str, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[bytes], str]:
        """按 msg_id 倒序找最近一张图。

        图文常常是两条独立消息（先发图、再说「这是你吗」），所以必须往前看几条。
        先取已经标出「这条有图」的，再逐条试剩下的（防标记失真）。
        """

        ordered = list(reversed(messages))
        for message in ordered:
            if not message.get("has_image"):
                continue
            data = await self._image_by_id(stream_id, str(message["id"]))
            if data:
                return data, str(message["id"])
        for message in ordered:
            if message.get("has_image"):
                continue
            data = await self._image_by_id(stream_id, str(message["id"]))
            if data:
                return data, str(message["id"])
        return None, ""


def create_plugin() -> LanyanSelfCheckPlugin:
    """Runner 通过此工厂函数实例化插件（缺了它插件会加载失败）。"""

    return LanyanSelfCheckPlugin()


__all__ = ["LanyanSelfCheckPlugin", "create_plugin"]
