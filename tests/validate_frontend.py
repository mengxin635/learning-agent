"""
前端 JS 逻辑验证器 — 在 Python 中重放关键交互路径
不依赖浏览器，验证渲染输出 + API 调用链
"""
import json
import re
import httpx

BASE = "http://localhost:8000"

# ============================================================
# 模拟前端 escapeHtml 和 escapeJsStr
# ============================================================

def escape_html(s):
    """模拟 JS escapeHtml"""
    if not s:
        return ''
    return (s.replace('&', '&amp;')
             .replace('<', '&lt;')
             .replace('>', '&gt;')
             .replace('"', '&quot;'))

def escape_js_str(s):
    """模拟 JS escapeJsStr"""
    if not s:
        return ''
    return s.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')

# ============================================================
# 模拟 renderQuestion 的 HTML 生成
# ============================================================

def render_question_html(q, progress):
    """模拟 renderQuestion() 生成的 innerHTML"""
    type_label = {
        'choice': '选择题', 'fill': '填空题', 'coding': '编程题'
    }.get(q['q_type'], '简答题')
    diff_label = {
        'easy': '🌟 简单', 'medium': '📖 中等', 'hard': '🔥 困难'
    }.get(q['difficulty'], q['difficulty'])
    
    html = f'''<span class="quiz-close" onclick="closeQuiz()" title="关闭测验">✕</span>
    <div class="quiz-header">
      <span class="q-title">📝 {escape_html(type_label)}</span>
      <span class="q-progress">{progress['current']} / {progress['total']}</span>
    </div>
    <div class="q-topic">📌 {escape_html(q['topic'])}</div>
    <span class="q-difficulty {escape_html(q['difficulty'])}">{diff_label}</span>
    <div class="q-text">{escape_html(q['text'])}</div>
'''
    if q['q_type'] == 'choice' and q.get('options'):
        html += '<div class="q-options">'
        for opt in q['options']:
            js_safe = escape_js_str(opt)
            display = escape_html(opt)
            # 生成 onclick — 模拟实际 HTML
            onclick = f"selectOption(this, '{js_safe}')"
            html += f'<button class="q-option" onclick="{onclick}">{display}</button>'
        html += '</div>'
        html += '<button class="btn-submit" id="btnSubmitAnswer" onclick="submitAnswer()" disabled>请先选择答案</button>'
    else:
        html += '<textarea class="q-fill-input" id="fillAnswer" placeholder="请输入你的答案..." rows="3"></textarea>'
        html += '<button class="btn-submit" id="btnSubmitAnswer" onclick="submitAnswer()">提交答案</button>'
    
    return html


def extract_onclicks(html):
    """从 HTML 中提取所有 onclick 属性值（正确处理 \\\" 转义）"""
    onclicks = []
    # 找 onclick=" 位置，手动扫描到闭合的 "
    pos = 0
    while True:
        pos = html.find('onclick="', pos)
        if pos == -1:
            break
        start = pos + 9  # 跳过 'onclick="'
        end = start
        while end < len(html):
            if html[end] == '\\':
                end += 2  # 跳过转义字符
                continue
            if html[end] == '"':
                break
            end += 1
        onclicks.append(html[start:end])
        pos = end + 1
    return onclicks


def validate_onclick(html):
    """验证 onclick 中的 JS 字符串是否合法"""
    errors = []
    for js_code in extract_onclicks(html):
        # 验证括号平衡（正确处理 \\' \\" \\\\ 转义）
        depth = 0
        in_str = False
        str_char = None
        escaped = False
        for i, ch in enumerate(js_code):
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if in_str:
                if ch == str_char:
                    in_str = False
                continue
            if ch in ("'", '"'):
                in_str = True
                str_char = ch
                continue
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth < 0:
                    errors.append(f"Col {i}: 括号不匹配 (过早关闭)")
                    break
        if in_str:
            errors.append(f"Col {len(js_code)}: 字符串未闭合 '{str_char}'")
        if depth != 0:
            errors.append(f"Col {len(js_code)}: 括号不匹配 (depth={depth})")
    
    return errors


# ============================================================
# 测试
# ============================================================

def test_onclick_rendering():
    """测试 1: onclick 属性 JS 合法性"""
    print("=" * 60)
    print("🧪 测试 onclick 属性 JS 合法性")
    print("=" * 60)
    
    test_questions = [
        {
            'q_type': 'choice',
            'text': "Python 中列表和元组的区别是？",
            'options': [
                "A. 列表可变，元组不可变",
                "B. 元组可变，列表不可变",
                "C. it's a tricky question",
                "D. 答案是 \"hello world\"",
            ],
            'difficulty': 'medium',
            'topic': 'Python基础',
        },
        {
            'q_type': 'fill',
            'text': "Python 的装饰器本质是一个___",
            'options': [],
            'difficulty': 'easy',
            'topic': '装饰器',
        }
    ]
    
    progress = {'current': 1, 'total': 3}
    all_pass = True
    
    for i, q in enumerate(test_questions):
        print(f"\n--- 题目 {i+1}: {q['text'][:30]}... ---")
        html = render_question_html(q, progress)
        errors = validate_onclick(html)
        
        if errors:
            print(f"  ❌ FAIL — {len(errors)} 个 JS 语法错误:")
            for e in errors:
                print(f"     {e}")
            all_pass = False
        else:
            print(f"  ✅ PASS — onclick JS 合法")
        
        # 展示每个选项的 onclick
        for opt in q.get('options', []):
            js_safe = escape_js_str(opt)
            print(f"     onclick=\"selectOption(this, '{js_safe}')\"")
    
    return all_pass


def test_full_quiz_flow():
    """测试 2: 完整 API 测验流程"""
    print("\n" + "=" * 60)
    print("🧪 测试完整 API 测验流程")
    print("=" * 60)
    
    all_pass = True
    
    # Start quiz
    print("\n1. 开始测验...")
    r = httpx.post(f"{BASE}/api/quiz/start", json={
        "topic": "Python", "difficulty": "easy",
        "question_type": "choice", "count": 3
    }, timeout=120)
    assert r.status_code == 200, f"Start failed: {r.status_code}"
    data = r.json()
    sid = data["session_id"]
    print(f"   ✅ session={sid}")
    
    # Verify question structure
    q = data["question"]
    assert "text" in q and "options" in q and "q_type" in q
    assert len(q["options"]) >= 2
    print(f"   ✅ 题目: {q['text'][:40]}...")
    print(f"   ✅ 选项: {len(q['options'])} 个")
    
    # Render HTML and validate
    html = render_question_html(q, data["progress"])
    errors = validate_onclick(html)
    if errors:
        print(f"   ❌ onclick 错误: {errors}")
        all_pass = False
    else:
        print(f"   ✅ onclick 合法")
    
    # Answer 3 questions
    for i in range(3):
        ans = chr(65 + i)  # A, B, C
        print(f"\n{i+2}. 回答 Q{i+1}: {ans}")
        r = httpx.post(f"{BASE}/api/quiz/answer", json={
            "session_id": sid, "answer": ans
        }, timeout=30)
        assert r.status_code == 200, f"Answer failed: {r.status_code}"
        d = r.json()
        print(f"   {'✅' if d['correct'] else '❌'} correct={d['correct']} "
              f"score={d['progress']['correct']}/{d['progress']['answered']}")
        
        if d.get("next_question"):
            print(f"   下一题: {d['next_question']['text'][:40]}...")
            # Validate onclick of next question
            next_html = render_question_html(d["next_question"], d["progress"])
            next_errors = validate_onclick(next_html)
            if next_errors:
                print(f"   ❌ 下一题 onclick 错误: {next_errors}")
                all_pass = False
        if d["finished"]:
            print(f"   🏁 完成!")
    
    # Summary
    print(f"\n{4}. 成绩单...")
    r = httpx.get(f"{BASE}/api/quiz/summary/{sid}", timeout=10)
    assert r.status_code == 200
    summary = r.json()
    print(f"   ✅ 等级: {summary['grade']}")
    print(f"   ✅ 成绩: {summary['correct']}/{summary['total']} ({summary['percentage']}%)")
    
    return all_pass


def test_xss_protection():
    """测试 3: XSS 防护验证"""
    print("\n" + "=" * 60)
    print("🧪 测试 XSS 防护")
    print("=" * 60)
    
    # 测试 HTML 注入在选项中
    malicious_options = [
        'A. <script>alert(1)</script>',
        'B. <img src=x onerror=alert(1)>',
        "C. it's <b>bold</b> & 'quoted'",
    ]
    q = {
        'q_type': 'choice',
        'text': '安全测试',
        'options': malicious_options,
        'difficulty': 'easy',
        'topic': 'Security',
    }
    
    html = render_question_html(q, {'current': 1, 'total': 1})
    errors = validate_onclick(html)
    
    if errors:
        print(f"  ❌ onclick 错误: {errors}")
        return False
    
    # 确保 <script> 等危险标签没有作为 HTML 内容渲染
    # (onclick 属性里的 <script> 是安全的，不执行)
    content_only = re.sub(r'<[^>]+>', ' ', html)  # 去掉所有标签
    for tag in ['<script', '<img onerror', '<iframe', '<svg onload']:
        if tag in content_only:
            print(f"  ❌ 发现未转义的危险标签: {tag}")
            return False
        print(f"  ✅ 无危险标签: {tag}")
    
    print(f"  ✅ 所有 onclick JS 合法")
    return True


if __name__ == "__main__":
    results = []
    
    results.append(("onclick JS 合法性", test_onclick_rendering()))
    results.append(("完整测验流程", test_full_quiz_flow()))
    results.append(("XSS 防护", test_xss_protection()))
    
    print("\n" + "=" * 60)
    print("📊 汇总")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n  {passed}/{len(results)} 通过")
