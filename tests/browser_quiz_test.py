"""
模拟鼠标浏览器测试 — 学习 Agent 测验功能
用 Playwright 模拟真实用户操作：点击、选择、提交
"""
import asyncio
import json
import sys
from pathlib import Path

# 确保能够 import
sys.path.insert(0, "/mnt/e/HermesProjects/learning-agent")

TEST_URL = "http://localhost:8000"

async def test_quiz_flow():
    from playwright.async_api import async_playwright
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        
        # 收集控制台错误
        console_errors = []
        page.on("console", lambda msg: 
            console_errors.append(f"[{msg.type}] {msg.text}") if msg.type == "error" else None
        )
        page.on("pageerror", lambda err: console_errors.append(f"[PAGE] {err}"))
        
        results = []
        
        # ==========================================
        # Test 1: 页面加载
        # ==========================================
        print("1️⃣  加载页面...")
        await page.goto(TEST_URL, wait_until="networkidle")
        await page.wait_for_timeout(1000)
        
        title = await page.title()
        header_text = await page.text_content("header .logo")
        results.append(("页面加载", title == "学习 Agent v1.4 · AI Tutor + Quiz" and "学习 Agent" in header_text))
        print(f"   标题: {title} | 头部: {header_text}")
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        # ==========================================
        # Test 2: 点击测验按钮打开侧边栏
        # ==========================================
        print("\n2️⃣  点击「📝 测验」按钮...")
        quiz_btn = await page.wait_for_selector("#btnQuiz", timeout=5000)
        await quiz_btn.click()
        await page.wait_for_timeout(500)
        
        sidebar_visible = await page.evaluate(
            "document.getElementById('sidebar').classList.contains('open')"
        )
        quiz_tab_visible = await page.evaluate(
            "document.getElementById('tabQuiz').classList.contains('active')"
        )
        results.append(("打开测验面板", sidebar_visible and quiz_tab_visible))
        print(f"   侧边栏打开: {sidebar_visible} | 测验标签页活动: {quiz_tab_visible}")
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        # ==========================================
        # Test 3: 填写表单并点击「开始测验」
        # ==========================================
        print("\n3️⃣  填写测验配置并开始...")
        await page.fill("#quizTopic", "Python")
        await page.select_option("#quizDifficulty", "easy")
        await page.select_option("#quizType", "choice")
        await page.fill("#quizCount", "3")
        
        start_btn = await page.wait_for_selector("#btnStartQuiz", timeout=5000)
        await start_btn.click()
        
        # 等待出题(最多60秒)
        try:
            await page.wait_for_selector(".quiz-overlay.show", timeout=60000)
            overlay_visible = True
        except:
            overlay_visible = False
        
        results.append(("开始测验弹窗", overlay_visible))
        print(f"   弹窗出现: {overlay_visible}")
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        if not overlay_visible:
            print("   ❌ 测验弹窗未出现，终止测试")
            print(f"\n   控制台错误: {console_errors}")
            await browser.close()
            return results, console_errors
        
        # ==========================================
        # Test 4: 检查题目渲染
        # ==========================================
        print("\n4️⃣  检查题目渲染...")
        q_text = await page.text_content(".quiz-modal .q-text")
        q_topic = await page.text_content(".quiz-modal .q-topic")
        q_progress = await page.text_content(".quiz-modal .q-progress")
        options = await page.query_selector_all(".quiz-modal .q-option")
        close_btn = await page.query_selector(".quiz-modal .quiz-close")
        
        print(f"   题目: {q_text[:60]}...")
        print(f"   主题: {q_topic}")
        print(f"   进度: {q_progress}")
        print(f"   选项数: {len(options)}")
        print(f"   关闭按钮: {'有' if close_btn else '无'}")
        
        results.append(("题目内容渲染", bool(q_text) and len(options) >= 2))
        results.append(("关闭按钮存在", close_btn is not None))
        print(f"   {'✅ PASS' if results[-2][1] else '❌ FAIL'} (内容) | {'✅ PASS' if results[-1][1] else '❌ FAIL'} (关闭)")
        
        # ==========================================
        # Test 5: 点击选项 — 模拟鼠标
        # ==========================================
        print("\n5️⃣  点击第一个选项...")
        first_option = options[0]
        option_text = await first_option.text_content()
        await first_option.click()
        await page.wait_for_timeout(300)
        
        is_selected = await first_option.evaluate("el => el.classList.contains('selected')")
        submit_enabled = await page.evaluate(
            "document.getElementById('btnSubmitAnswer').disabled === false"
        )
        selected_option = await page.evaluate("window.selectedOption")
        
        results.append(("选项高亮", is_selected))
        results.append(("提交按钮启用", submit_enabled))
        print(f"   选项文本: {option_text}")
        print(f"   高亮: {is_selected} | 按钮启用: {submit_enabled} | 选中值: {selected_option}")
        print(f"   {'✅ PASS' if results[-2][1] else '❌ FAIL'} (高亮) | {'✅ PASS' if results[-1][1] else '❌ FAIL'} (按钮)")
        
        if not is_selected or not submit_enabled:
            print("   ❌ 选项点击无响应 — onclick 转义仍有问题!")
        
        # ==========================================
        # Test 6: 提交答案
        # ==========================================
        print("\n6️⃣  点击「提交答案」...")
        submit_btn = await page.wait_for_selector("#btnSubmitAnswer", timeout=5000)
        await submit_btn.click()
        
        # 等待判卷结果
        try:
            await page.wait_for_selector(".result-overlay.show", timeout=30000)
            result_visible = True
        except:
            result_visible = False
        
        results.append(("判卷结果显示", result_visible))
        print(f"   结果弹窗: {result_visible}")
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        if result_visible:
            result_icon = await page.text_content(".result-toast .result-icon")
            result_text = await page.text_content(".result-toast .result-text")
            print(f"   判卷: {result_icon} | {result_text}")
        
        # ==========================================
        # Test 7: 点击「下一题」继续
        # ==========================================
        print("\n7️⃣  点击「下一题」...")
        next_btn = await page.wait_for_selector(".btn-next", timeout=5000)
        btn_label = await next_btn.text_content()
        await next_btn.click()
        await page.wait_for_timeout(500)
        
        # 检查是否进入了下一题
        overlay_still_visible = await page.evaluate(
            "document.getElementById('quizOverlay').classList.contains('show')"
        )
        new_q_text = await page.text_content(".quiz-modal .q-text") if overlay_still_visible else ""
        
        results.append(("下一题加载", overlay_still_visible and bool(new_q_text)))
        print(f"   按钮文字: {btn_label}")
        print(f"   弹窗仍在: {overlay_still_visible} | 新题目: {new_q_text[:60] if new_q_text else 'N/A'}...")
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        # ==========================================
        # Test 8: 答完所有题
        # ==========================================
        print("\n8️⃣  继续答完剩余题目...")
        for i in range(2):  # 还有2题
            # 点击第一个选项
            opts = await page.query_selector_all(".quiz-modal .q-option")
            if opts:
                await opts[0].click()
                await page.wait_for_timeout(200)
            
            # 提交
            submit = await page.wait_for_selector("#btnSubmitAnswer", timeout=5000)
            await submit.click()
            
            try:
                await page.wait_for_selector(".result-overlay.show", timeout=30000)
            except:
                pass
            
            # 下一题或查看成绩单
            nxt = await page.wait_for_selector(".btn-next", timeout=5000)
            await nxt.click()
            await page.wait_for_timeout(500)
        
        # ==========================================
        # Test 9: 成绩单
        # ==========================================
        print("\n9️⃣  检查成绩单...")
        summary_card = await page.query_selector(".summary-card")
        if summary_card:
            grade = await page.text_content(".summary-card .s-grade")
            score = await page.text_content(".summary-card .s-score")
            print(f"   等级: {grade} | 成绩: {score}")
            results.append(("成绩单显示", True))
        else:
            results.append(("成绩单显示", False))
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        # ==========================================
        # Test 10: 关闭按钮
        # ==========================================
        print("\n🔟 测试关闭按钮...")
        # 重新开始一个快速测验
        await page.fill("#quizCount", "2")
        start_btn2 = await page.wait_for_selector("#btnStartQuiz", timeout=5000)
        await start_btn2.click()
        try:
            await page.wait_for_selector(".quiz-overlay.show", timeout=60000)
        except:
            pass
        
        close = await page.query_selector(".quiz-modal .quiz-close")
        if close:
            # page.on("dialog") 处理确认框
            await page.click(".quiz-modal .quiz-close")
            await page.wait_for_timeout(500)
            
            # 检查弹窗是否关闭
            overlay_closed = not await page.evaluate(
                "document.getElementById('quizOverlay').classList.contains('show')"
            )
            results.append(("关闭按钮功能", overlay_closed))
            print(f"   弹窗关闭: {overlay_closed}")
        else:
            results.append(("关闭按钮功能", False))
            print(f"   关闭按钮不存在")
        print(f"   {'✅ PASS' if results[-1][1] else '❌ FAIL'}")
        
        # ==========================================
        # 总结
        # ==========================================
        print("\n" + "="*60)
        print("📊 测试结果汇总")
        print("="*60)
        passed = 0
        for name, ok in results:
            status = "✅ PASS" if ok else "❌ FAIL"
            print(f"  {status}  {name}")
            if ok: passed += 1
        
        print(f"\n  {passed}/{len(results)} 通过")
        
        if console_errors:
            print(f"\n⚠️  控制台错误 ({len(console_errors)}):")
            for e in console_errors[:10]:
                print(f"  {e}")
        
        await browser.close()
        return results, console_errors


if __name__ == "__main__":
    results, errors = asyncio.run(test_quiz_flow())
    all_pass = all(r[1] for r in results)
    sys.exit(0 if all_pass else 1)
