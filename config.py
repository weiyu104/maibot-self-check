"""自像比对 (lanyan.self-check) · 配置模型

只做一件事：图 + 自我指向文本时，把「待判图」和「她的标准三视图」一起喂给视觉模型，
把比对结论作为一条材料注入 planner 的请求尾部。

🔴 本插件不产出聊天消息、不写数据库、不改图片描述——只改这一轮 planner 的请求。
"""

from typing import List

from maibot_sdk import Field, PluginConfigBase


class PluginSection(PluginConfigBase):
    """插件开关"""

    __ui_label__ = "插件"
    __ui_order__ = 0

    name: str = Field(default="lanyan_self_check", description="插件名称", json_schema_extra={"label": "插件标识", "order": 0, "hint": "内部 ID，请勿修改", "hidden": True, "disabled": True})
    version: str = Field(default="0.2.4", description="插件版本", json_schema_extra={"label": "插件版本", "order": 1, "hint": "由插件维护，请勿修改", "hidden": True, "disabled": True})
    config_version: str = Field(default="1.0.0", description="配置版本", json_schema_extra={"label": "配置版本", "order": 2, "hint": "由插件维护，请勿修改", "hidden": True, "disabled": True})
    enabled: bool = Field(default=True, description="是否启用", json_schema_extra={"label": "启用自像比对", "order": 3, "hint": "总开关", "description": "关闭后不再发起比对，不影响其它插件与聊天功能"})


class TriggerSection(PluginConfigBase):
    """什么时候才比对（B 方案：有图 + 自我指向文本）"""

    __ui_label__ = "触发条件"
    __ui_order__ = 1

    require_image: bool = Field(
        default=True,
        description="必须带图才比对。关掉会变成每条消息都比（费钱且没意义），不建议关",
        json_schema_extra={"label": "必须带图才比对", "order": 0, "hint": "🔴 建议保持开启：关掉会变成每条消息都比对，既费钱又没意义"},
    )
    keywords: List[str] = Field(
        default_factory=lambda: [
            "像你",
            "是你",
            "画你",
            "你自己",
            "你本人",
            "你的样子",
            "自画像",
            "你长什么样",
            "这就是你",
            "和你一样",
        ],
        description="自我指向短语（子串匹配）。命中任意一条且带图才触发比对；只放比「你」更长的词，避免「你看这个」也触发",
        json_schema_extra={"label": "触发短语", "order": 1, "hint": "逗号分隔、子串匹配。只放比「你」更长的词，避免「你看这个」误触发", "placeholder": "例：像你, 是你, 自画像, 你长什么样"},
    )
    lookback: int = Field(
        default=8,
        ge=1,
        le=20,
        description=(
            "往前看几条用户消息找图。图文常是两条独立消息（先发图再说「这是你吗」），"
            "群里消息密，图很容易被后面几句挤出小窗口，所以默认 8 条"
        ),
        json_schema_extra={"label": "找图回溯条数", "order": 2, "hint": "图文常分两条消息发送；群里消息密，默认 8 条"},
    )
    group_require_at: bool = Field(
        default=True,
        description=(
            "群聊里必须 @ 了她、或者直接叫了她的名字/别名，才比对。"
            "判定用宿主算好的 is_at / is_mentioned（昵称与别名在 bot_config 的 [bot] 段）。"
            "关掉的话，群友之间随口一句「是你吗」也会拿别人的图去判"
        ),
        json_schema_extra={"label": "群聊需 @ 或点名", "order": 3, "hint": "避免群友随口一句「是你吗」拿别人的图去判"},
    )
    text_lookback: int = Field(
        default=4,
        ge=1,
        le=20,
        description=(
            "找「这是你吗」这类问句时最多往回看几条（取图仍按 lookback 找）。"
            "只取最后一条文本会因为后一句「在？」把问句顶掉，但不该无限往前翻"
        ),
        json_schema_extra={"label": "问句回溯条数", "order": 4, "hint": "找「这是你吗」这类问句时最多往回看几条；取图仍按上一项"},
    )
    text_max_age_seconds: int = Field(
        default=120,
        ge=0,
        le=1800,
        description=(
            "问句与最新一条消息的最大间隔（秒），0 = 不限。"
            "实测「这是你吗」与随后的「在？」相隔 84 秒，设 60 秒会正好漏掉这个场景，"
            "所以默认 120"
        ),
        json_schema_extra={"label": "问句有效时长（秒）", "order": 5, "hint": "0 = 不限。默认 120 秒，覆盖「先发图、过一会儿再问」"},
    )


class EngineSection(PluginConfigBase):
    """视觉模型直连参数"""

    __ui_label__ = "比对引擎"
    __ui_order__ = 2

    model: str = Field(
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        description="视觉模型。需支持一次请求带两张图",
        json_schema_extra={"label": "视觉模型", "order": 0, "hint": "需支持一次请求带两张图", "placeholder": "Qwen/Qwen3-VL-30B-A3B-Instruct"},
    )
    provider_name: str = Field(
        default="SiliconFlow",
        description="从 model_config.toml 里按这个名字取 base_url / api_key；下面两项填了就优先用下面的",
        json_schema_extra={"label": "模型配置名称", "order": 1, "hint": "从宿主 model_config.toml 按这个名字取地址与密钥"},
    )
    base_url: str = Field(
        default="",
        description="留空 = 自动读 model_config.toml 的 SiliconFlow 接口地址",
        json_schema_extra={"label": "接口地址", "order": 2, "hint": "留空 = 自动读取宿主模型配置", "placeholder": "留空即可"},
    )
    api_key: str = Field(
        default="",
        description="留空 = 自动读 model_config.toml 里的密钥（推荐留空，别把密钥抄到插件配置里）",
        json_schema_extra={"label": "API 密钥", "order": 3, "hint": "🔴 建议留空，自动读取宿主密钥；不要把密钥写进插件配置", "placeholder": "留空 = 自动读取", "secret": True},
    )
    timeout_seconds: float = Field(
        default=15.0,
        gt=3.0,
        le=120.0,
        description="单次比对超时。实测 4.5~10.8 秒，15 秒留约 1.4 倍余量；超时就放弃本轮注入，不重试（重试等于让她等两倍）",
        json_schema_extra={"label": "单次超时（秒）", "order": 4, "hint": "超时就放弃本轮注入，不重试（重试会让人等两倍时间）", "advanced": True, "step": 0.1},
    )
    max_ref_side: int = Field(
        default=1280,
        ge=384,
        le=2048,
        description="参考图缩放后的最长边。1024 实测够认发饰，再大只涨 token",
        json_schema_extra={"label": "基准图最长边（px）", "order": 5, "hint": "太小会糊掉发饰等细节，太大只增加 token", "advanced": True},
    )
    max_test_side: int = Field(
        default=1024,
        ge=384,
        le=2048,
        description="待判图缩放后的最长边",
        json_schema_extra={"label": "待判图最长边（px）", "order": 6, "hint": "缩放后送检的边长", "advanced": True},
    )
    jpeg_quality: int = Field(default=85, ge=50, le=95, description="缩放后 JPEG 质量", json_schema_extra={"label": "JPEG 质量", "order": 7, "hint": "50~95，越高越清晰也越费 token", "advanced": True})
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="采样温度，判别任务要低", json_schema_extra={"label": "采样温度", "order": 8, "hint": "判别任务建议保持低值", "advanced": True, "step": 0.1})
    max_tokens: int = Field(default=300, ge=80, le=800, description="输出上限，够 JSON 就行", json_schema_extra={"label": "输出上限", "order": 9, "hint": "够返回一段 JSON 即可", "advanced": True})


class RefsSection(PluginConfigBase):
    """参考池"""

    __ui_label__ = "参考图"
    __ui_order__ = 3

    primary: str = Field(
        default="refs/three_view.jpg",
        description="主基准图。相对路径按插件目录解析；默认是正/侧/背三视图",
        json_schema_extra={"label": "主基准图", "order": 0, "hint": "相对插件目录的路径；换成自己的角色三视图即可复用本插件", "placeholder": "refs/three_view.jpg"},
    )
    fallback: str = Field(
        default="",
        description="主基准图缺失时的备用图。留空 = 不启用备用图，请自备基准图",
        json_schema_extra={"label": "备用基准图", "order": 1, "hint": "主图缺失时使用；留空则不启用备用图", "placeholder": "留空 = 不启用备用图"},
    )


class LimitsSection(PluginConfigBase):
    """额度与缓存"""

    __ui_label__ = "额度与缓存"
    __ui_order__ = 4

    daily_max: int = Field(
        default=80,
        ge=0,
        description="每日比对上限（0 = 不限）。请按所用模型的实际单价估算开销",
        json_schema_extra={"label": "每日比对上限", "order": 0, "hint": "0 = 不限。按你所用模型的实际单价估算开销"},
    )
    cache_minutes: int = Field(
        default=30,
        ge=0,
        description="同一张图（按字节 hash）的结论缓存多久。0 = 不缓存",
        json_schema_extra={"label": "同图缓存（分钟）", "order": 1, "hint": "同一张图按字节 hash 缓存结论，0 = 不缓存"},
    )


class InjectSection(PluginConfigBase):
    """注入文案"""

    __ui_label__ = "注入文案"
    __ui_order__ = 5

    template: str = Field(
        default=("[自像比对] 已比对：{verdict_cn}，置信 {confidence}。{diff_line}{advice}"),
        description=(
            "注入到 planner 请求尾部的一行。可用占位符：{verdict_cn} {verdict} {confidence} "
            "{differences} {diff_line} {one_line} {advice}（advice 是按结论自动生成的说话要求）"
        ),
        json_schema_extra={"label": "注入模板", "order": 0, "hint": "占位符：{verdict_cn} {verdict} {confidence} {differences} {diff_line} {one_line} {advice}", "advanced": True},
    )
    verdict_labels: str = Field(
        default="same=同一个角色（换了衣服/场景/画风也算同一个）|similar=很像，但外貌有出入|different=不是同一个角色|unclear=没法判断",
        description="四档结论的中文说法，用 | 分隔、= 对应",
        json_schema_extra={"label": "四档结论的中文名", "order": 1, "hint": "用 | 分隔、= 对应，例如 same=同一个角色|similar=很像", "advanced": True},
    )


class PromptSection(PluginConfigBase):
    """比对提示词"""

    __ui_label__ = "比对提示词"
    __ui_order__ = 6

    compare: str = Field(
        default="""图1是【你的角色名】的三视图参考（正面/侧面/背面，同一个角色）。图2是待判断的图片。

🔴 图2里可能有一个或多个人物（双人同框、合影、和别的角色一起出现都很常见）。
比对前先在心里做这件事（不要把过程写出来）：在图2里逐个找出所有人物，分别记下每个人的
发色、发型、瞳色；然后拿【每一个人】分别跟图1比。只要其中有一个人三条主特征对得上，
结论就是 same。
不要因为同框的另一个人不是本人、或者另一个人的发色更醒目、画在更中间，就判 different；
也不要把别人的发色发型按到本人身上。判断时以「像不像图1这个角色」为准，不是「这张图整体像不像」。

这是 AI 生成的图或同人图，细节常有偏差。先看下面三条「主特征」，主特征对得上就判 same，
不要因为细节瑕疵（配饰有无、颜色浓淡、瞳色冷暖）就降级。

主特征（必须对得上；⚠️ 下面三行请换成你自己角色的外貌，否则判据无意义）：
① 发色发型：【用一句话写清发色、长度、刘海，例如「栗色及肩卷发、斜刘海」】
② 瞳色：【例如：偏蓝紫，紫中带蓝、有高光】
③ 脸型：【例如：下巴偏尖，细眉长眼】
【另写一行说明什么算「本质不同」，例如：整头彩发、短发、卷发 = 不同】

参考项（不作为降级依据，只帮确认）：【例如：常戴的发饰；不同造型可能不戴，一律不影响判级】。

不算差异的：服装、鞋子、发饰与配饰、姿势、动作、背景、光线、画风（含 Q 版/厚涂/赛璐璐）、
构图、表情、年龄感，以及三条主特征的浓淡深浅。

判级：
same      = 三条主特征都对得上（细节有偏差照样算 same）
similar   = 有 1 项主特征明显不对，但整体仍是同一角色
different = 主特征本质不同，或明显是另一个人
unclear   = 看不清、被遮挡、或无法判断

differences 只写【本人】与图1的外貌差异（最多 3 条，每条 12 字内），不要写服装、背景、动作，
也不要把同框另一个人的长相当成本人的差异。判 same 时一般留空。
one_line 是给聊天用的一句话结论，20 字内，第三人称陈述，例如「是同一个人，只是换了衣服」；
图里有别人时要点明，例如「是我，旁边那个不是我」。

严格输出 JSON，不要别的内容：
{"verdict":"same|similar|different|unclear","confidence":0.0,"differences":[],"one_line":""}""",
        description="两图比对提示词。改动热重载即生效",
        json_schema_extra={"label": "两图比对提示词", "order": 0, "hint": "只写判据，不重述身份细节；身份一致性由基准图承载。改动热重载即生效", "placeholder": "按主特征判级、严格输出 JSON 的说明文字", "advanced": True},
    )


class DebugSection(PluginConfigBase):
    """诊断（平时留空）"""

    __ui_label__ = "诊断"
    __ui_order__ = 9

    probe_msg_id: str = Field(
        default="",
        description="填一个消息 ID，插件加载时会试着取这条消息的图并打日志。验证完请清空，平时不要填",
        json_schema_extra={"label": "诊断消息 ID", "order": 0, "hint": "平时留空；填入后插件加载时会尝试取这条消息的图并打日志", "placeholder": "平时留空", "advanced": True},
    )
    trace: bool = Field(
        default=False,
        description="每轮规划都打一行诊断（拿到几条消息、最后一句是什么、关键词有没有命中）。排查完关掉，平时别开",
        json_schema_extra={"label": "逐轮诊断日志", "order": 1, "hint": "排查完请关闭，平时别开", "advanced": True},
    )

class SelfCheckConfig(PluginConfigBase):
    """自像比对配置"""

    __ui_label__ = "自像比对"
    __ui_order__ = 0

    plugin: PluginSection = Field(default_factory=PluginSection)
    trigger: TriggerSection = Field(default_factory=TriggerSection)
    engine: EngineSection = Field(default_factory=EngineSection)
    refs: RefsSection = Field(default_factory=RefsSection)
    limits: LimitsSection = Field(default_factory=LimitsSection)
    inject: InjectSection = Field(default_factory=InjectSection)
    prompt: PromptSection = Field(default_factory=PromptSection)
    debug: DebugSection = Field(default_factory=DebugSection)
