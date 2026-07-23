"""
微博网络语言词典和常量定义
包含微博特有的情感表达词汇和语义映射
"""

from typing import Dict, List, Set, Any, Optional
from enum import Enum


class WeiboLanguageType(Enum):
    """微博语言类型枚举"""
    FAN_CIRCLE = "粉圈语言"  # 粉丝话语体系
    SPORTS = "体育赛事"     # 体育相关表达
    TRENDING = "网络热词"   # 网络流行语
    EMOJI = "表情符号"      # 表情情感表达
    SARCASM = "讽刺表达"    # 讽刺幽默表达


class WeiboConstants:
    """微博网络语言常量定义"""

    # 正向情感词汇（加权词汇）
    POSITIVE_WORDS: Set[str] = {
        # 粉圈正向表达
        '打call', '应援', '打call', '吹爆', '绝了', '神仙', '封神', '好看', '爱了',
        '宝藏', '上头', '入坑', '安利', '种草', '拔草', '种草', '上头', '太上头',
        '神仙颜值', '惊艳', '心动', '磕到了', '上头了', '绝绝子', 'yyds', 'YYDS',

        # 体育正向表达
        '给力', '牛逼', '王者', '冠军', '胜利', '夺冠', '赢麻了', '赢麻',
        '冠军相', '碾压', '无敌', '霸气', '统治力', '教科书', '经典',

        # 网络热词正向
        '笑死', '哈哈哈哈', '笑不活了', '666', '厉害了', '棒棒哒', '太棒了',
        '完美', '爱了爱了', '赞', '点赞', '收藏', 'mark', '学习了',
        '感恩', '感谢', '好人一生平安', '谢谢', '辛苦', '支持',

        # 程度加强表达
        '超级', '特别', '非常', '极其', '十分', '太', '超级无敌', '炸裂',
        '无敌', '神级', '顶级', '巅峰', '天花板', '极致', '完美',
    }

    # 负向情感词汇（降权词汇）
    NEGATIVE_WORDS: Set[str] = {
        # 粉圈负向表达
        '下头', '烂', '辣鸡', '吐了', '恶心', '恶心心', '难看', '失望',
        '脱粉', '圈钱', '割韭菜', '恰饭', '恰烂钱', '恰烂饭', '无语',
        '迷惑行为', '迷惑', '迷惑操作', '迷惑发言', '迷惑行为大赏',

        # 体育负向表达
        '输麻了', '输麻', '崩了', '崩溃', '心态崩了', '心态爆炸',
        '垃圾', '菜', '菜狗', '菜鸡', '菜逼', '废物', '废物点心',
        '拉胯', '拉跨', '拉爆', '爆炸', '垫底', '副班长', '鱼腩',

        # 网络热词负向
        '绷不住了', '离谱', '离大谱', '太离谱', '就离谱', '真的离谱',
        '逆天', '太逆天', '纯纯逆天', '逆天了', '逆大天',
        '吐了', '真吐了', '想吐', '恶心', '难崩', '难绷',
        '蚌埠住了', '蚌埠住', '蚌不住了', '无语子', '无语住了',

        # 程度加强表达
        '太', '超级', '特别', '非常', '极其', '十分',
        '垃圾中的战斗机', '菜的一批', '菜的抠脚', '菜的惊天动地',
    }

    # 反问模式规则
    RHETORICAL_PATTERNS: List[str] = [
        r'难道.*不',
        r'怎么.*会',
        r'谁.*会',
        r'你觉得.*吗\?',
        r'那.*为什么',
        r'这不.*吗\?',
        r'这也.*吧\?',
        r'哪里(？！.*(有|在|是|可以|能)).*\?',
        r'.*在哪\??',
        r'.*什么(好|值得|应该).*\?',
    ]

    # 粉圈语言特征词
    FAN_CIRCLE_KEYWORDS: Set[str] = {
        '打call', '应援', '站姐', '后援会', '粉丝团', '超话', '签到',
        '控评', '轮博', '打投', '做数据', '空瓶', '卡黑', '反黑',
        '产出', '画手', '写手', '剪辑', '二创', '同人', '粮',
        '本命', '墙头', '爱豆', '偶像', '蒸煮', '正主',
    }

    # 体育语言特征词
    SPORTS_KEYWORDS: Set[str] = {
        '比分', '进球', '射门', '进球', '得分', '篮板', '助攻',
        '冠军', '亚军', '季军', '淘汰', '晋级', '小组赛', '决赛',
        '教练', '队员', '阵容', '战术', '策略', '训练', '比赛',
        '球迷', '主场', '客场', '门票', '转播', '解说', '直播',
    }

    # 表情符号情感映射
    EMOJI_SENTIMENT: Dict[str, float] = {
        # 正向表情
        '😂': 0.8, '😊': 0.7, '😍': 0.9, '🥰': 0.9, '😎': 0.6,
        '👍': 0.7, '👏': 0.8, '🎉': 0.9, '💕': 0.8, '✨': 0.6,
        '🌟': 0.7, '🔥': 0.8, '💯': 0.9, '💪': 0.7, '🙏': 0.5,

        # 负向表情
        '😭': -0.8, '😔': -0.6, '😞': -0.7, '😩': -0.8, '😟': -0.5,
        '😠': -0.8, '😡': -0.9, '💔': -0.9, '😫': -0.7, '🙄': -0.4,

        # 中性表情
        '😐': 0.0, '🤔': -0.1, '😶': 0.0, '😏': -0.2,
    }

    # 网络用语语义映射
    TRENDING_SEMANTICS: Dict[str, Dict[str, Any]] = {
        # 正向用语语义
        '绝绝子': {'sentiment': 0.9, 'type': WeiboLanguageType.TRENDING, 'description': '极致赞美'},
        'yyds': {'sentiment': 0.95, 'type': WeiboLanguageType.TRENDING, 'description': '永远的神'},
        '打call': {'sentiment': 0.8, 'type': WeiboLanguageType.FAN_CIRCLE, 'description': '支持欢呼'},
        '笑死': {'sentiment': 0.6, 'type': WeiboLanguageType.TRENDING, 'description': '非常有趣'},
        '666': {'sentiment': 0.7, 'type': WeiboLanguageType.TRENDING, 'description': '厉害佩服'},

        # 负向用语语义
        '离谱': {'sentiment': -0.8, 'type': WeiboLanguageType.TRENDING, 'description': '难以置信的糟糕'},
        '逆天': {'sentiment': -0.9, 'type': WeiboLanguageType.TRENDING, 'description': '超出理解的糟糕'},
        '绷不住了': {'sentiment': -0.6, 'type': WeiboLanguageType.TRENDING, 'description': '难以忍受'},
        '下头': {'sentiment': -0.7, 'type': WeiboLanguageType.FAN_CIRCLE, 'description': '失望反感'},
        '吐了': {'sentiment': -0.8, 'type': WeiboLanguageType.TRENDING, 'description': '极度厌恶'},

        # 中性或需要具体分析的
        '啊这': {'sentiment': 0.0, 'type': WeiboLanguageType.TRENDING, 'description': '语境依赖'},
        '就这': {'sentiment': -0.3, 'type': WeiboLanguageType.SARCASM, 'description': '质疑或讽刺'},
    }

    @classmethod
    def get_emoji_sentiment_score(cls, emoji: str) -> float:
        """获取表情符号情感得分"""
        return cls.EMOJI_SENTIMENT.get(emoji, 0.0)

    @classmethod
    def get_trending_sentiment_score(cls, word: str) -> Optional[float]:
        """获取网络用语情感得分"""
        semantic = cls.TRENDING_SEMANTICS.get(word)
        return semantic['sentiment'] if semantic else None

    @classmethod
    def detect_language_type(cls, text: str) -> List[WeiboLanguageType]:
        """检测文本的语言类型"""
        types = set()

        # 检测粉圈语言
        if any(keyword in text for keyword in cls.FAN_CIRCLE_KEYWORDS):
            types.add(WeiboLanguageType.FAN_CIRCLE)

        # 检测体育语言
        if any(keyword in text for keyword in cls.SPORTS_KEYWORDS):
            types.add(WeiboLanguageType.SPORTS)

        # 检测网络热词
        if any(trending in text for trending in cls.TRENDING_SEMANTICS.keys()):
            types.add(WeiboLanguageType.TRENDING)

        # 检测表情符号
        if any(emoji in text for emoji in cls.EMOJI_SENTIMENT.keys()):
            types.add(WeiboLanguageType.EMOJI)

        return list(types)

    @classmethod
    def is_fan_circle_text(cls, text: str) -> bool:
        """判断是否为粉圈文本"""
        return WeiboLanguageType.FAN_CIRCLE in cls.detect_language_type(text)

    @classmethod
    def is_sports_text(cls, text: str) -> bool:
        """判断是否为体育文本"""
        return WeiboLanguageType.SPORTS in cls.detect_language_type(text)

    @classmethod
    def contains_trending_words(cls, text: str) -> bool:
        """判断是否包含网络热词"""
        return WeiboLanguageType.TRENDING in cls.detect_language_type(text)