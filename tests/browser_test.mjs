const { chromium } = require('playwright');

(async () => {
  const results = [];
  const errors = [];
  const URL = 'http://localhost:8000';

  console.log('🧪 学习 Agent 测验功能 — Edge 浏览器鼠标模拟测试\n');

  // 使用 Windows 上的 Edge 浏览器（Chromium 内核）
  const browser = await chromium.launch({ 
    headless: true,
    executablePath: '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  page.on('console', msg => { if (msg.type() === 'error') errors.push(`[console] ${msg.text()}`); });
  page.on('pageerror', err => errors.push(`[page] ${err.message}`));

  try {
    // ====== 1. 加载页面 ======
    console.log('1️⃣  加载页面...');
    await page.goto(URL, { waitUntil: 'networkidle' });
    const header = await page.textContent('header .logo');
    console.log(`   头部: ${header}`);
    results.push({ name: '页面加载', ok: header.includes('学习 Agent') });

    // ====== 2. 点击测验按钮 ======
    console.log('\n2️⃣  点击「📝 测验」按钮...');
    await page.click('#btnQuiz');
    await page.waitForTimeout(500);
    const sidebarOpen = await page.evaluate(() => 
      document.getElementById('sidebar').classList.contains('open')
    );
    const quizTabActive = await page.evaluate(() =>
      document.getElementById('tabQuiz').classList.contains('active')
    );
    results.push({ name: '打开测验面板', ok: sidebarOpen && quizTabActive });
    console.log(`   侧边栏: ${sidebarOpen} | 标签页: ${quizTabActive}`);

    // ====== 3. 配置并开始测验 ======
    console.log('\n3️⃣  填写配置并点击「开始测验」...');
    await page.fill('#quizTopic', 'Python');
    await page.selectOption('#quizDifficulty', 'easy');
    await page.selectOption('#quizType', 'choice');
    await page.fill('#quizCount', '3');
    
    await page.click('#btnStartQuiz');
    console.log('   等待出题...');
    
    try {
      await page.waitForSelector('.quiz-overlay.show', { timeout: 60000 });
      console.log('   ✅ 测验弹窗已出现');
      results.push({ name: '开始测验弹窗', ok: true });
    } catch (e) {
      console.log('   ❌ 测验弹窗未出现 (60s超时)');
      results.push({ name: '开始测验弹窗', ok: false });
      // 检查是否有错误消息
      const chatText = await page.textContent('#chat');
      console.log(`   聊天区最后内容: ${chatText.slice(-200)}`);
      await browser.close();
      printResults(results, errors);
      return;
    }

    // ====== 4. 检查题目渲染 ======
    console.log('\n4️⃣  检查题目渲染...');
    const qText = await page.textContent('.quiz-modal .q-text');
    const qTopic = await page.textContent('.quiz-modal .q-topic');
    const qProgress = await page.textContent('.quiz-modal .q-progress');
    const options = await page.$$('.quiz-modal .q-option');
    const closeBtn = await page.$('.quiz-modal .quiz-close');
    
    console.log(`   题目: ${qText?.slice(0, 60)}...`);
    console.log(`   主题: ${qTopic}`);
    console.log(`   进度: ${qProgress}`);
    console.log(`   选项数: ${options.length}`);
    console.log(`   关闭按钮: ${closeBtn ? '有' : '无'}`);
    results.push({ name: '题目渲染', ok: !!qText && options.length >= 2 });
    results.push({ name: '关闭按钮', ok: !!closeBtn });

    // ====== 5. 点击第一个选项（模拟鼠标） ======
    console.log('\n5️⃣  鼠标点击第一个选项...');
    const opt = options[0];
    const optText = await opt.textContent();
    console.log(`   选项文字: ${optText}`);
    
    await opt.click();
    await page.waitForTimeout(300);
    
    const isSelected = await opt.evaluate(el => el.classList.contains('selected'));
    const submitEnabled = await page.evaluate(() => 
      document.getElementById('btnSubmitAnswer')?.disabled === false
    );
    const selectedVal = await page.evaluate(() => window.selectedOption);
    
    console.log(`   高亮: ${isSelected} | 按钮启用: ${submitEnabled} | selectedOption = "${selectedVal}"`);
    results.push({ name: '选项点击高亮', ok: isSelected });
    results.push({ name: '提交按钮启用', ok: submitEnabled });
    
    if (!isSelected || !submitEnabled) {
      console.log('   ❌ 选项点击无响应！onclick 可能仍有问题');
    }

    // ====== 6. 提交答案 ======
    console.log('\n6️⃣  点击「提交答案」...');
    await page.click('#btnSubmitAnswer');
    
    try {
      await page.waitForSelector('.result-overlay.show', { timeout: 30000 });
      console.log('   ✅ 判卷结果已出现');
      const resultIcon = await page.textContent('.result-toast .result-icon');
      const resultText = await page.textContent('.result-toast .result-text');
      console.log(`   判卷: ${resultIcon} | ${resultText}`);
      results.push({ name: '判卷结果', ok: true });
    } catch (e) {
      console.log('   ❌ 判卷结果未出现');
      results.push({ name: '判卷结果', ok: false });
    }

    // ====== 7. 下一题 ======
    console.log('\n7️⃣  点击「下一题」继续...');
    const nextBtn = await page.$('.btn-next');
    const btnLabel = await nextBtn?.textContent();
    console.log(`   按钮: ${btnLabel}`);
    
    if (nextBtn) {
      await nextBtn.click();
      await page.waitForTimeout(500);
      
      const stillVisible = await page.evaluate(() =>
        document.getElementById('quizOverlay').classList.contains('show')
      );
      const newQ = stillVisible ? await page.textContent('.quiz-modal .q-text') : '';
      console.log(`   弹窗仍在: ${stillVisible} | 新题目: ${newQ?.slice(0, 50)}...`);
      results.push({ name: '下一题加载', ok: stillVisible && !!newQ });
    }

    // ====== 8. 继续答完所有题 ======
    console.log('\n8️⃣  继续答完剩余题目...');
    for (let i = 0; i < 2; i++) {
      const opts = await page.$$('.quiz-modal .q-option');
      if (opts.length > 0) {
        await opts[0].click();
        await page.waitForTimeout(200);
      }
      const submit = await page.$('#btnSubmitAnswer');
      if (submit) {
        await submit.click();
        try { await page.waitForSelector('.result-overlay.show', { timeout: 30000 }); } catch {}
      }
      const nxt = await page.$('.btn-next');
      if (nxt) {
        await nxt.click();
        await page.waitForTimeout(300);
      }
    }
    console.log('   ✅ 所有题目已完成');

    // ====== 9. 成绩单 ======
    console.log('\n9️⃣  检查成绩单...');
    await page.waitForTimeout(500);
    const grade = await page.textContent('.summary-card .s-grade');
    const score = await page.textContent('.summary-card .s-score');
    console.log(`   等级: ${grade} | 成绩: ${score}`);
    results.push({ name: '成绩单', ok: !!grade && !!score });

    // ====== 10. 无 JS 错误 ======
    if (errors.length === 0) {
      console.log('\n🔟 无控制台错误 ✅');
      results.push({ name: '无 JS 错误', ok: true });
    } else {
      console.log(`\n🔟 控制台错误: ${errors.length} 个`);
      for (const e of errors) console.log(`   ${e}`);
      results.push({ name: '无 JS 错误', ok: false });
    }

  } catch (e) {
    console.log(`\n❌ 测试异常: ${e.message}`);
    results.push({ name: '测试异常', ok: false, detail: e.message });
  }

  await browser.close();
  printResults(results, errors);
})();

function printResults(results, errors) {
  console.log('\n' + '='.repeat(60));
  console.log('📊 测试结果汇总');
  console.log('='.repeat(60));
  let passed = 0;
  for (const r of results) {
    const icon = r.ok ? '✅' : '❌';
    console.log(`  ${icon} ${r.name}`);
    if (r.ok) passed++;
  }
  console.log(`\n  ${passed}/${results.length} 通过`);
  if (errors.length > 0) {
    console.log(`\n⚠️  控制台错误:`);
    for (const e of errors) console.log(`  ${e}`);
  }
  process.exit(passed === results.length ? 0 : 1);
}
