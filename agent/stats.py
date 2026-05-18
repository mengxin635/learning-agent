"""
学习统计模块
为仪表盘提供数据聚合：总览、趋势、知识点分布
"""
from datetime import datetime, timedelta
from typing import Dict, List
from .memory import memory


def get_overview() -> Dict:
    """学习总览：掌握度、测验成绩、待复习数"""
    kg = memory.get_knowledge_summary()
    mem_status = memory.get_status()

    # 测验数据
    quiz_total = 0
    quiz_correct = 0
    try:
        from .quiz import QUIZ_SESSIONS_DIR
        import os, json
        if os.path.exists(QUIZ_SESSIONS_DIR):
            for fname in os.listdir(QUIZ_SESSIONS_DIR):
                if fname.endswith(".json"):
                    with open(os.path.join(QUIZ_SESSIONS_DIR, fname)) as f:
                        data = json.load(f)
                    quiz_total += data.get("total", 0)
                    quiz_correct += data.get("score", 0)
    except Exception:
        pass

    return {
        "total_topics": kg.get("total_topics", 0),
        "mastered_topics": kg.get("mastered", 0),
        "learning_topics": kg.get("learning", 0),
        "due_review": kg.get("due_review", 0),
        "avg_mastery": kg.get("avg_mastery", 0),
        "quiz_total_questions": quiz_total,
        "quiz_correct": quiz_correct,
        "quiz_accuracy": round(quiz_correct / quiz_total * 100, 1) if quiz_total > 0 else 0,
        "short_term_messages": mem_status.get("short_term", 0),
        "long_term_memories": mem_status.get("long_term", 0),
    }


def get_trend(days: int = 30) -> List[Dict]:
    """每日学习趋势：最近 N 天的活动量"""
    daily_activity = {}
    today = datetime.now().date()

    # 测验活动
    try:
        from .quiz import QUIZ_SESSIONS_DIR
        import os, json
        if os.path.exists(QUIZ_SESSIONS_DIR):
            for fname in os.listdir(QUIZ_SESSIONS_DIR):
                if fname.endswith(".json"):
                    with open(os.path.join(QUIZ_SESSIONS_DIR, fname)) as f:
                        data = json.load(f)
                    created = data.get("created_at", "")
                    if created:
                        date_str = created[:10]
                        daily_activity[date_str] = daily_activity.get(date_str, 0) + 1
    except Exception:
        pass

    # 知识图谱学习记录
    for node in memory.kg.nodes.values():
        if node.last_study:
            date_str = node.last_study[:10]
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1

    trend = []
    for i in range(days - 1, -1, -1):
        date = (today - timedelta(days=i)).isoformat()
        trend.append({
            "date": date,
            "activity": daily_activity.get(date, 0),
        })

    return trend


def get_topic_distribution() -> Dict:
    """知识点掌握度分布"""
    topics = memory.kg.get_all_topics()
    distribution = {
        "mastered": [],
        "learning": [],
        "weak": [],
        "labels": [],
        "values": [],
    }

    for t in topics:
        m = t.get("mastery", 0)
        name = t.get("topic", "")
        distribution["labels"].append(name)
        distribution["values"].append(round(m * 100, 1))

        if m >= 0.65:
            distribution["mastered"].append({"name": name, "mastery": round(m * 100)})
        elif m >= 0.35:
            distribution["learning"].append({"name": name, "mastery": round(m * 100)})
        else:
            distribution["weak"].append({"name": name, "mastery": round(m * 100)})

    return distribution
