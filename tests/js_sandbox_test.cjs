/**
 * 精确验证 escapeJsStr — 检查每个字符的 charCode
 */
// === 从页面复制的 escapeJsStr ===
function escapeJsStr(s) {
  if (!s) return '';
  return s.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '\\"');
}

function escapeHtml(s) {
  if (!s) return '';
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
          .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ====== 测试用例 ======
const tests = [
  "A. plain text",
  "B. it's a test",
  'C. "hello" world',
  "D. <script>xss</script>",
  "A. 列表可变，元组不可变",
];

let allPass = true;

for (const input of tests) {
  const escaped = escapeJsStr(input);
  
  // 模拟 renderQuestion 中的模板字面量
  const onclickAttr = `onclick="selectOption(this, '${escaped}')"`;
  
  console.log(`\n输入: ${JSON.stringify(input)}`);
  console.log(`转义: ${JSON.stringify(escaped)}`);
  console.log(`onclick: ${onclickAttr}`);
  
  // 模拟浏览器行为: 
  // 1. HTML 解析器提取属性值（去掉外层引号）
  const attrMatch = onclickAttr.match(/onclick="(.+)"/);
  const attrValue = attrMatch ? attrMatch[1] : '';
  
  // 2. JS 引擎执行 onclick 中的代码 — 提取字符串参数
  // selectOption(this, 'ESCAPED')
  const argMatch = attrValue.match(/selectOption\(this,\s*'((?:[^'\\]|\\.)*)'\)/);
  
  if (!argMatch) {
    console.log(`  ❌ 无法从 onclick 提取参数`);
    console.log(`     attrValue: ${attrValue}`);
    allPass = false;
    continue;
  }
  
  const jsStringLiteral = argMatch[1]; // 含转义序列的 JS 字符串字面量
  
  // 3. JS 引擎解码转义序列
  let decoded = '';
  for (let i = 0; i < jsStringLiteral.length; i++) {
    if (jsStringLiteral[i] === '\\' && i + 1 < jsStringLiteral.length) {
      const next = jsStringLiteral[i + 1];
      if (next === '\\' || next === "'" || next === '"') {
        decoded += next;
        i++;
      } else {
        decoded += '\\' + next;
        i++;
      }
    } else {
      decoded += jsStringLiteral[i];
    }
  }
  
  const ok = decoded === input;
  console.log(`  JS字面量: ${JSON.stringify(jsStringLiteral)}`);
  console.log(`  解码结果: ${JSON.stringify(decoded)}`);
  console.log(`  ${ok ? '✅' : '❌'} 往返: ${decoded === input ? '正确' : '错误!'}`);
  
  if (!ok) allPass = false;
  
  // 额外检查: onclick HTML 中是否有 ' 未转义
  // 检查 onclick 属性值里，去掉转义的 \' 后，是否还有裸引号
  const bareQuotes = attrValue.replace(/\\'/g, '').match(/'/g);
  if (bareQuotes) {
    // 应该刚好2个（参数两边的引号）
    if (bareQuotes.length !== 2) {
      console.log(`  ❌ onclick 属性中引号不平衡: ${bareQuotes.length} 个`);
      allPass = false;
    }
  }
}

// ====== XSS 验证 ======
console.log('\n=== XSS ===');
const xssInput = '<script>alert(1)</script>';
const xssOutput = escapeHtml(xssInput);
console.log(`输入: ${xssInput}`);
console.log(`输出: ${xssOutput}`);
console.log(`安全: ${!xssOutput.includes('<script>') ? '✅' : '❌'}`);

// ====== 完整模拟 ======
console.log('\n=== 完整 onclick 生成 ===');
const options = ["A. it's fine", 'B. "hello"', "C. normal"];
for (const opt of options) {
  const jsSafe = escapeJsStr(opt);
  const htmlSafe = escapeHtml(opt);
  const button = `<button class="q-option" onclick="selectOption(this, '${jsSafe}')">${htmlSafe}</button>`;
  console.log(button);
}

console.log(`\n${'='.repeat(50)}`);
console.log(allPass ? '✅ 全部通过' : '❌ 有失败');
process.exit(allPass ? 0 : 1);
